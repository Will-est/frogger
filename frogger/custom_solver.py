import numpy as np
import time
from typing import List, Tuple, Optional

from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

import nlopt

from frogger.custom_robot_model import FunctionalRobotModel

class FunctionalFrogger(Frogger):
    """Frogger solver with functional contact constraints."""
    
    def __init__(
        self, 
        cfg: FroggerConfig,
        actuation_contacts: List[Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> None:
        # if not isinstance(cfg.model, FunctionalRobotModel):
        #     raise TypeError("Model must be FunctionalRobotModel")
        
        super().__init__(cfg)
        self.model.actuation_contacts = actuation_contacts

    def compute_contact_correspondence(self, fingertip_poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
        """Compute correspondence between fingertips and functional contacts."""
        fingertip_positions = np.array([pose[0] for pose in fingertip_poses])
        n_fingers = len(fingertip_poses)
        n_functional = len(self.model.actuation_contacts)
        
        # Compute distance matrix, any palm related contact distance should be set to inf
        distances = np.zeros((n_fingers, n_functional))
        for i, (pos, _) in enumerate(fingertip_poses):
            is_palm = "palm" in self.model.fingertip_names[i]
            for j, (func_pos, _) in enumerate(self.model.actuation_contacts):
                is_palm = is_palm or "palm" in self.model.fingertip_names[j]
                if is_palm:
                    distances[i, j] = np.inf
                else:
                    distances[i, j] = np.linalg.norm(pos - func_pos)
        
        # Assign correspondences
        correspondence = [-1] * n_fingers
        assigned = set()
        
        for _ in range(n_functional):
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if j not in assigned:
                correspondence[i] = j
                assigned.add(j)
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            
        return correspondence

    def generate_grasp(self, optimize=True, check_constraints=False, timeout_seconds: Optional[float] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate grasp with functional contacts. If timeout_seconds is provided,
        the grasp generation loop will abort if the total elapsed time exceeds timeout_seconds.
        """
        start_total = time.time()
        success = False
        sampling_times = []
        optimization_times = []
        
        while not success:
            # Check if overall timeout has been exceeded.
            if timeout_seconds is not None and (time.time() - start_total) > timeout_seconds:
                raise TimeoutError(f"generate_grasp timed out after {timeout_seconds} seconds")
            
            # Time sampling
            sampling_start = time.time()
            q0, _ = self.sampler.sample_configuration(**kwargs)
            sampling_time = time.time() - sampling_start
            sampling_times.append(sampling_time)

            # Compute correspondence and update model
            self.model.set_q(q0)
            fingertip_poses = self.model.compute_fingertip_poses()
            self.model.contact_correspondence = self.compute_contact_correspondence(fingertip_poses)
            
            if not optimize:
                if check_constraints:
                    success = self.check_constraints(q0)
                    if success:
                        return None, q0
                    else:
                        continue
                else:
                    return None, q0            

            # Time optimization
            optimization_start = time.time()
            try:
                q_star = self.opt.optimize(q0)
                optimization_time = time.time() - optimization_start
                optimization_times.append(optimization_time)
            except (ValueError, nlopt.RoundoffLimited) as e:
                print(f"Optimization error: {e}! Resampling...")
                q_star = np.nan * np.ones(self.model.n)
            except RuntimeError as e:
                raise e
            except Exception as e:
                raise e

            # Check solution
            if np.any(np.isnan(q_star)):
                print("Failed: Optimization returned NaN values")
                continue
                    
            # Verify constraints
            g_val = np.zeros(self.n_ineq)
            self.g(g_val, q_star, np.zeros(0))
            h_val = np.zeros(self.n_eq)
            self.h(h_val, q_star, np.zeros(0))

            # Compute constraint violations
            surf_vio = np.max(np.abs(h_val[: self.n_surf]))
            couple_vio = np.max(np.abs(h_val[self.n_surf:(self.n_surf + self.n_couple)])) if self.model.n_couple > 0 else 0.0
            h_extra_vio = np.max(np.abs(h_val[(self.n_surf + self.n_couple):])) if len(h_val[(self.n_surf + self.n_couple):]) > 0 else 0.0
                
            joint_vio = max(np.max(g_val[: self.model.n_bounds]), 0.0)
            col_vio = max(np.max(g_val[self.model.n_bounds:(self.model.n_bounds + self.model.n_pairs)]), 0.0)
            g_extra_vio = max(np.max(g_val[(self.model.n_bounds + self.model.n_pairs):]), 0.0) if len(g_val[(self.model.n_bounds + self.model.n_pairs):]) > 0 else 0.0

            # Print specific failure reasons
            if surf_vio > self.tol_surf:
                print(f"Failed: Surface contact constraint violation ({surf_vio:.2e} > {self.tol_surf})")
            if couple_vio > self.tol_couple:
                print(f"Failed: Coupling constraint violation ({couple_vio:.2e} > {self.tol_couple})")
            if joint_vio > self.tol_joint:
                print(f"Failed: Joint limit violation ({joint_vio:.2e} > {self.tol_joint})")
            if col_vio > self.tol_col:
                print(f"Failed: Collision constraint violation ({col_vio:.2e} > {self.tol_col})")
            if g_extra_vio > self.tol_fclosure or h_extra_vio > self.tol_fclosure:
                print(f"Failed: Force closure constraint violation (g:{g_extra_vio:.2e}, h:{h_extra_vio:.2e} > {self.tol_fclosure})")

            # Check if solution is feasible
            success = (
                surf_vio <= self.tol_surf and
                couple_vio <= self.tol_couple and
                joint_vio <= self.tol_joint and
                col_vio <= self.tol_col and
                g_extra_vio <= self.tol_fclosure and
                h_extra_vio <= self.tol_fclosure
            )
                
            if success:
                # Print timing statistics
                print("Success!")
                print("\nTiming Statistics:")
                print("Sampling:")
                print(f"  Total attempts: {len(sampling_times)}")
                print(f"  Total time: {sum(sampling_times):.3f}s")
                print(f"  Average time: {np.mean(sampling_times):.3f}s")
                print(f"  Std deviation: {np.std(sampling_times):.3f}s")
                print(f"  Last attempt: {sampling_times[-1]:.3f}s")
                    
                print("\nOptimization:")
                print(f"  Total attempts: {len(optimization_times)}")
                print(f"  Total time: {sum(optimization_times):.3f}s")
                print(f"  Average time: {np.mean(optimization_times):.3f}s")
                print(f"  Std deviation: {np.std(optimization_times):.3f}s")
                print(f"  Last attempt: {optimization_times[-1]:.3f}s")
                    
                print("\nConstraints:")
                print(f"Surface: {surf_vio:.2e}, Couple: {couple_vio:.2e}, Joint: {joint_vio:.2e}, Collision: {col_vio:.2e}, Extra: {g_extra_vio:.2e}")
                return q_star, q0
            

    def check_constraints(self, q):
        self.model.set_q(q)
        
        # Verify constraints
        g_val = np.zeros(self.n_ineq)
        self.g(g_val, q, np.zeros(0))
        h_val = np.zeros(self.n_eq)
        self.h(h_val, q, np.zeros(0))

        # Check constraint violations
        surf_vio = np.max(np.abs(h_val[: self.n_surf]))
        couple_vio = np.max(np.abs(h_val[self.n_surf : (self.n_surf + self.n_couple)])) if self.model.n_couple > 0 else 0.0
        h_extra_vio = np.max(np.abs(h_val[(self.n_surf + self.n_couple):])) if len(h_val[(self.n_surf + self.n_couple):]) > 0 else 0.0
        
        joint_vio = max(np.max(g_val[: self.model.n_bounds]), 0.0)
        col_vio = max(np.max(g_val[self.model.n_bounds : (self.model.n_bounds + self.model.n_pairs)]), 0.0)
        g_extra_vio = max(np.max(g_val[(self.model.n_bounds + self.model.n_pairs):]), 0.0) if len(g_val[(self.model.n_bounds + self.model.n_pairs):]) > 0 else 0.0

        # Print specific failure reasons
        if surf_vio > self.tol_surf:
            print(f"Failed: Surface contact constraint violation ({surf_vio:.2e} > {self.tol_surf})")
        if couple_vio > self.tol_couple:
            print(f"Failed: Coupling constraint violation ({couple_vio:.2e} > {self.tol_couple})")
        if joint_vio > self.tol_joint:
            print(f"Failed: Joint limit violation ({joint_vio:.2e} > {self.tol_joint})")
        if col_vio > self.tol_col:
            print(f"Failed: Collision constraint violation ({col_vio:.2e} > {self.tol_col})")
        if g_extra_vio > self.tol_fclosure or h_extra_vio > self.tol_fclosure:
            print(f"Failed: Force closure constraint violation (g:{g_extra_vio:.2e}, h:{h_extra_vio:.2e} > {self.tol_fclosure})")

        # Check if solution is feasible
        success = (
            surf_vio <= self.tol_surf
            and couple_vio <= self.tol_couple
            and joint_vio <= self.tol_joint
            and col_vio <= self.tol_col
            and g_extra_vio <= self.tol_fclosure
            and h_extra_vio <= self.tol_fclosure
        )
