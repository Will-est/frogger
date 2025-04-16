import numpy as np
from typing import Optional, List, Tuple
from pydrake.math import RigidTransform

from frogger.robots.robot_core import RobotModel, RobotModelConfig
from frogger.robots.robots import AlgrModel, AlgrModelConfig
from frogger.robots.custom_robots import LeapModel, LeapModelConfig
from frogger.grasping import (
    compute_gOCs,
    compute_grasp_matrix,
    compute_primitive_forces,
    wedge,
)
from frogger.metrics import min_weight_gradient, min_weight_lp

def get_fingertip_names(cfg: RobotModelConfig) -> list[str]:
    if isinstance(cfg, AlgrModelConfig):
        hand = cfg.hand
        if cfg.palm_contact:
            return sorted([
                f"algr_{hand}_if_ds",
                f"algr_{hand}_mf_ds",
                f"algr_{hand}_palm",
                f"algr_{hand}_rf_ds",
                f"algr_{hand}_th_ds",
            ])
        else:
            return sorted([
                f"algr_{hand}_if_ds",
                f"algr_{hand}_mf_ds",
                f"algr_{hand}_rf_ds",
                f"algr_{hand}_th_ds",
            ])
    elif isinstance(cfg, LeapModelConfig):
        hand = cfg.hand
        if cfg.palm_contact:
            return sorted([
                f"leap_{hand}_if_ds",
                f"leap_{hand}_mf_ds",
                f"leap_{hand}_palm",
                f"leap_{hand}_rf_ds",
                f"leap_{hand}_th_ds",
            ])
        else:
            return sorted([
                f"leap_{hand}_if_ds",
                f"leap_{hand}_mf_ds",
                f"leap_{hand}_rf_ds",
                f"leap_{hand}_th_ds",
            ])
    
def get_offsets(cfg: RobotModelConfig) -> list[np.ndarray]:
    # Contact point offsets in the fingertip frame
    if isinstance(cfg, AlgrModelConfig):
        th_t = np.pi / 4.0
        r_f = 0.012
        if cfg.palm_contact:
            fingertip_offsets = [
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # index
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # middle
                np.array([0.0, 0.0, 0.0]),  # palm
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # ring
                np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)]),  # thumb
            ]
            return fingertip_offsets
        else:
            fingertip_offsets = [
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # index
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # middle
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # ring
                np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)],)  # thumb
            ]
            return fingertip_offsets
    else:
        fingertip_offsets = [
            np.array([-0.011, -0.035, 0.01318]) for _ in range(3)
        ] + [np.array([-0.009, -0.048, -0.01128])]
        return fingertip_offsets
    
def custom_fingertip_height_constraints(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce that the z-coordinate of each fingertip is non-negative.
    Returns a tuple (g_extra, Dg_extra) where:
      - g_extra[i] = -p_z[i]   (which must be ≤ 0)
      - Dg_extra[i] is the gradient of -p_z[i] with respect to q.
    """
    # Compute the fingertip poses; each pose is (position, rotation matrix)
    poses = model.compute_fingertip_poses()
    num_tips = len(poses)
    
    # Initialize the extra inequality constraint vector and its gradient
    g_extra = np.empty(num_tips)
    Dg_extra = np.empty((num_tips, model.n))
    
    # Make sure the fingertip Jacobians are up-to-date
    # (Assuming compute_fingertip_poses has also updated model.J_tips)
    for i, (p, _) in enumerate(poses):
        # Constraint: -p_z <= 0
        g_extra[i] = -p[2]
        # The gradient is simply - (the third row of the fingertip Jacobian)
        Dg_extra[i, :] = -model.J_tips[i][2, :]
        
    return g_extra, Dg_extra

class FunctionalRobotModel(AlgrModel):
    """Robot model that supports required contact points."""
    
    def __init__(self, cfg: RobotModelConfig) -> None:
        super().__init__(cfg)
        # Initialize attributes for actuation contacts
        self.actuation_contacts: Optional[List[Tuple[np.ndarray, Optional[np.ndarray]]]] = None 
        self.contact_correspondence: Optional[List[int]] = None
        self.actuation_constraint_type: str = "fingertip"
        self.fingertip_names = get_fingertip_names(cfg)
        self.fingertip_offsets = get_offsets(cfg)
        self.normal_contact_weight = 1e-1

        self.reverse_actuation = isinstance(cfg, LeapModelConfig)
    
    # def compute_Df(self, q: np.ndarray) -> np.ndarray:
    #     df = super().compute_Df(q)
    #     df *= 1e-2
    #     return df

    def compute_fingertip_poses(self, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.plant_context is None:
            raise RuntimeError("Plant context not initialized")
            
        hand = self.hand
        # Get fingertip frames
        fingertip_frames = [
            self.plant.GetBodyByName(name).body_frame() for name in self.fingertip_names
        ]        

        if finger_idx is not None:
            frame = fingertip_frames[finger_idx]
            X_WF = self.plant.CalcRelativeTransform(
                self.plant_context,
                self.plant.world_frame(),
                frame
            )
            p_WC = self.plant.CalcPointsPositions(
                self.plant_context,
                frame,
                self.fingertip_offsets[finger_idx],
                self.plant.world_frame(),
            ).squeeze()
            return [(p_WC, X_WF.rotation().matrix())]

        fingertip_poses = []
        for frame, offset in zip(fingertip_frames, self.fingertip_offsets):
            X_WF = self.plant.CalcRelativeTransform(
                self.plant_context,
                self.plant.world_frame(),
                frame
            )
            p_WC = self.plant.CalcPointsPositions(
                self.plant_context,
                frame,
                offset,
                self.plant.world_frame(),
            ).squeeze()
            fingertip_poses.append((p_WC, X_WF.rotation().matrix()))
        return fingertip_poses
    
    def compute_contact_poses(self, q, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute contact poses by finding collision points."""
        q = q.copy()
        if self.plant_context is None:
            raise RuntimeError("Plant context not initialized")
            
        # Process collisions to get contact points if not already computed
        if self.p_tips is None:
            self._process_collisions(q)

        contact_poses = [(pos, normal) for pos, normal in zip(self.p_tips, self.compute_n_W(q))]
        if finger_idx is not None:
            return [contact_poses[finger_idx]]
        
        return contact_poses
    
    # def compute_fingertip_poses(self, finger_idx: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    #     """Compute fingertip poses by finding collision points."""
    #     if self.plant_context is None:
    #         raise RuntimeError("Plant context not initialized")
            
    #     # Process collisions to get fingertip points
    #     if self.p_tips is None:
    #         self._process_collisions(self.plant.GetPositions(self.plant_context, self.robot_instance))
        
    #     if finger_idx is not None:
    #         return [(self.p_tips[finger_idx], self.compute_n_W(self.q)[finger_idx])]
            
    #     return [(pos, normal) for pos, normal in zip(self.p_tips, self.compute_n_W(self.q))]
    
    def compute_normals(self, pos):
        """Compute normals for the given positions."""
        return self.obj.Ds_W(pos, batched=True)
    
    def _compute_G_and_W(self) -> None:
        """Computes grasp and wrench matrices, with minimal contribution from actuation contacts."""
        # Basic computations same as original 
        X_WO = self.obj.X_WO
        X_OW = X_WO.inverse()
        R_OW = X_OW.inverse().rotation().matrix()
        
        self.P_OF = X_OW @ self.p_tips.T  # (3, nc)
        Ds_p = self.obj.Ds_W(self.p_tips, batched=True)
        self.n_W = -Ds_p.T / np.linalg.norm(Ds_p, axis=1)
        self.n_O = R_OW @ self.n_W

        # Compute full matrices first
        self.gOCs = compute_gOCs(self.P_OF, self.n_O)
        self.G = compute_grasp_matrix(self.gOCs)
        self.W = self.G @ np.kron(np.eye(self.nc), self.F)

        # Scale down contribution of actuation contacts in W matrix
        # if self.contact_correspondence is not None:
        #     for i in range(self.nc):
        #         if self.contact_correspondence[i] != -1:
        #             # Scale the columns corresponding to this contact to be very small
        #             col_start = i * len(self.F) + 7
        #             col_end = (i + 1) * len(self.F) + 7
        #             self.W[:, col_start:col_end] *= 1e-3
        # else:
        #     Warning("Contact correspondence not set. Using full l matrix.")


    def _compute_l(self) -> None:
        """Computes the min-weight metric considering only grasp contacts."""
        super()._compute_l()
        # l, dl = self.l, self.Dl
        # self.l *= 1e-1
        # self.Dl *= 1e-1
        # if self.contact_correspondence is not None:
        #     for i in range(self.nc):
        #         if self.contact_correspondence[i] != -1:
        #             # Scale the columns corresponding to this contact to be very small
        #             col_start = i * len(self.F) + 7
        #             col_end = (i + 1) * len(self.F) + 7
        #             # l[col_start:col_end] *= 1e-3
        #             dl[col_start:col_end] *= 1e-3
        # else:
        #     Warning("Contact correspondence not set. Using full l matrix.")

  
    def _compute_eq_cons(self, q: np.ndarray) -> None:
        if self.h is None:
            self._init_eq_cons()
            
        # Pre-compute poses
        # Get positions based on mode
        # if self.actuation_constraint_type == "contact":
        #     current_poses = self.compute_contact_poses()
        # elif self.actuation_constraint_type == "fingertip":
        current_poses = self.compute_fingertip_poses()
        # current_poses = self.compute_contact_poses(q)
        
        # For each finger, we'll have one scalar constraint
        h_constraints = np.zeros(self.nc)
        Dh_constraints = np.zeros((self.nc, self.n))
        fingertip_names = self.fingertip_names
        contact_idx = []
        # for i in range(self.nc):

        
        max_distance = -np.inf
        for i in range(self.nc):
            fingertip_name = fingertip_names[i]
            if "palm" in fingertip_name:
                print("Skipping palm")
                continue

            if self.contact_correspondence and self.contact_correspondence[i] != -1:
                act_pos, act_dir = self.actuation_contacts[self.contact_correspondence[i]]
                current_pos, current_rot = current_poses[i]

                # act_pos = act_pos.copy()
                # act_pos -= act_dir * 0.03
                
                # Signed distance
                pos_error = np.linalg.norm(current_pos - act_pos)
                # determine the sign
                # sign = np.sign(np.dot(current_pos - act_pos, act_dir))
                # pos_error *= sign

                max_distance = max(max_distance, pos_error)
                h_constraints[i] = pos_error
                # h_constraints[i] = self.h_tip[i]
                # print(self.h_tip[i])
                error_direction = (current_pos - act_pos) / pos_error
                # error_direction = (act_pos - current_pos) / pos_error
                # if pos_error > 1e-10:
                #     error_direction = (current_pos - act_pos) / pos_error
                # else:
                #     error_direction = np.zeros(3)
                Dh_constraints[i] = error_direction @ self.J_tips[i]
                # Dh_constraints[i] = 0.8 * error_direction @ self.J_tips[i] + 0.2 * self.Dh_tip[i]
                
                # Add direction err
                # if False:
                if act_dir is not None:
                    if self.reverse_actuation:
                        act_dir = act_dir.copy() * -1.
                    # Use the x axis if that is desired; otherwise, change to the appropriate axis.
                    current_dir = current_rot[:, 0]  
                    # Ensure the dot product is in the valid range
                    dot_val = np.clip(np.dot(current_dir, act_dir), -1.0, 1.0)
                    
                    # Define a squared cost (zero when perfectly aligned)
                    angle_cost = 0.5 * (1 - dot_val)**2 * 5e-3  # scaling factor can be tuned
                    h_constraints[i] += angle_cost
                    
                    # Compute the gradient of the cost with respect to current_dir:
                    # d(angle_cost)/d(current_dir) = -5e-3 * (1 - dot_val) * act_dir
                    angle_grad = -5e-3 * (1 - dot_val) * act_dir
                    
                    # Propagate the gradient through the fingertip Jacobian:
                    Dh_constraints[i] += angle_grad @ self.J_tips[i]
            else:
                # h_constraints[i] = self.h_tip[i]
                # Dh_constraints[i] = self.Dh_tip[i]
                h_constraints[i] = self.h_tip[i] * self.normal_contact_weight
                Dh = self.Dh_tip[i]
                mask = np.ones_like(self.Dh_tip[i]) * self.normal_contact_weight
                # mask = np.ones_like(self.Dh_tip[i]) * 0.0
                mask[7+i*4:11+i*4] = 1
                Dh = Dh * mask
                Dh_constraints[i] = Dh
        # print(f"h constraints: {h_constraints}")
        # print(f"Dh constraints: {Dh_constraints}")
        # print(f"Max distance: {max_distance}")
        
        # Store constraints
        if self.n_couple != 0:
            h_couple = self.A_couple @ q + self.b_couple
            self.h[: self.n_couple + self.nc] = np.concatenate((h_constraints, h_couple))
            self.Dh[: self.n_couple + self.nc, :] = np.concatenate((Dh_constraints, self.A_couple))
        else:
            self.h[: self.nc] = h_constraints
            self.Dh[: self.nc, :] = Dh_constraints
        
        # Add any extra constraints
        if self.custom_compute_h is not None:
            h_extra, Dh_extra = self.custom_compute_h(self)
            assert len(h_extra) == self.n_h_extra
            self.h[self.n_couple + self.nc :] = h_extra
            self.Dh[self.n_couple + self.nc :, :] = Dh_extra