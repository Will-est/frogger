
from typing import Optional, Union
from dataclasses import dataclass

from frogger.sampling import HeuristicICSampler, HeuristicAlgrICSampler
from frogger.robots.robots import AlgrModel, RobotModel
from frogger.robots.custom_robots import LeapModel
from frogger.utils import make_rotation_for_direction, find_opposite_contacts, add_marker

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.geometry import Sphere, Rgba
from pydrake.visualization import AddFrameTriadIllustration


class ConfigurableHeuristicICSampler(HeuristicAlgrICSampler):
    """
    A configurable version of HeuristicICSampler that allows overriding specific behaviors.
    Inherits all default implementations from HeuristicICSampler.
    """
    def __init__(
        self,
        model: RobotModel,
        custom_palm_sampler: Optional['CustomPalmSampler'] = None,
        custom_constraint: Optional['CustomConstraint'] = None,
        table: bool = False,
        z_axis_fwd: bool = False
    ) -> None:
        super().__init__(model, table=table, z_axis_fwd=z_axis_fwd)
        self._custom_palm_sampler = custom_palm_sampler
        self._custom_constraint = custom_constraint

    def sample_palm_pose(self) -> RigidTransform:
        """
        Uses custom palm sampler if provided, otherwise falls back to parent implementation.
        """
        if self._custom_palm_sampler is not None:
            return self._custom_palm_sampler.sample_palm_pose(self)
        return super().sample_palm_pose()

    def add_additional_constraints(
        self,
        ik: InverseKinematics,
        X_WPalm_des: RigidTransform
    ) -> None:
        """
        Uses custom constraint if provided, otherwise falls back to parent implementation.
        """
        if self._custom_constraint is not None:
            return self._custom_constraint.add_additional_constraints(self, ik, X_WPalm_des)
        return super().add_additional_constraints(ik, X_WPalm_des)
    
    def add_visualization(self, model, path_prefix="sampled_grasp"):
        """
        Visualize contact points with the following color scheme:
        - Black: actuation contacts
        - Non-actuation contacts:
            - Red: index finger (0)
            - Green: middle finger (1)
            - Blue: ring finger (2)
            - Yellow: thumb (3)
        """
        # Delete any existing visualization
        try:
            model.meshcat.Delete(path=path_prefix)
        except:
            pass

        if self._custom_palm_sampler is not None:
            # Visualize palm pose
            add_marker(model, self.X_WPalm_des.translation(), color=[0.0, 1.0, 0.0, 1.0], radius=0.01, name=f"{path_prefix}/palm")

        # Color mapping for non-actuation fingers
        finger_colors = {
            0: [1.0, 0.0, 0.0, 1.0],  # Red for index
            1: [0.0, 1.0, 0.0, 1.0],  # Green for middle
            2: [0.0, 0.0, 1.0, 1.0],  # Blue for ring
            3: [1.0, 1.0, 0.0, 1.0],  # Yellow for thumb
            4: [1.0, 0.0, 1.0, 1.0],  # Magenta for pinky
        }

        fingertip_poses = model.compute_fingertip_poses()
        for i, (pos, ort) in enumerate(fingertip_poses):
            if hasattr(self.model, "act_assignments") and i in self.model.act_assignments:
                color = [0.0, 0.0, 0.0, 1.0]  # Black for actuation contacts
            else:
                color = finger_colors[i]
            sphere_name = f"{path_prefix}/fingertip_{i}"
            add_marker(model, pos, ort, color=color, name=sphere_name)            

        if self._custom_constraint is not None:
            # Visualize actuation contacts (black)
            for i, (pos, _) in self.model.act_assignments.items():
                sphere_name = f"{path_prefix}/actuation_contact_{i}"
                add_marker(model, pos, ort=None, color=[0.0, 0.0, 0.0, 1.0], name=sphere_name)

            # Visualize normal contacts (color-coded by finger)
            if hasattr(self._custom_constraint, 'normal_assignments'):
                obj = model.obj
                for i, pos_local in self._custom_constraint.normal_assignments.items():
                    pos_world = obj.X_WO @ pos_local
                    sphere_name = f"{path_prefix}/normal_contact_{i}"
                    add_marker(model, pos_world, color=finger_colors[i], name=sphere_name)


# Helper classes for custom behaviors
class CustomPalmSampler:
    """Base class for custom palm sampling strategies"""
    def sample_palm_pose(self, sampler: ConfigurableHeuristicICSampler) -> RigidTransform:
        raise NotImplementedError

class CustomConstraint:
    """Base class for custom constraint adding strategies"""
    def add_additional_constraints(
        self,
        sampler: ConfigurableHeuristicICSampler,
        ik: InverseKinematics,
        X_WPalm_des: RigidTransform
    ) -> None:
        raise NotImplementedError

# Example implementations
class ContactPalmSampler(CustomPalmSampler):
    """
    Implementation of palm sampling from PalmHeuristicAlgrICSampler
    Note: palm_offset can be a float, or a list for randomization range
    """
    def __init__(self, contact_predictor, palm_offset: Union[float,tuple] = (0.005, 0.05)):
        self.contact_predictor = contact_predictor
        self.palm_offset = palm_offset
        self.support_contacts = None

    def sample_palm_pose(self, sampler: ConfigurableHeuristicICSampler) -> RigidTransform:
        obj = sampler.model.obj
        heatmap, pts, normals = self.contact_predictor.query()
        vertices = np.array(pts)

        # Sample contact point
        sucess = False
        while not sucess:
            contact_idx = np.random.choice(len(heatmap), p=heatmap/heatmap.sum())
            contact_point_O = vertices[contact_idx]
            contact_point_W = obj.X_WO @ contact_point_O
            
            # Get corresponding normal
            normal_O = normals[contact_idx]
            normal_W = obj.X_WO.rotation().matrix() @ normal_O
            normal_W = normal_W / np.linalg.norm(normal_W)

            # Set x-axis opposite to surface normal (palm facing contact)
            x_hat = -normal_W

            # Select y-axis similar to original sampler
            X_WO = obj.X_WO
            X_OBB = obj.X_OBB
            axis_lengths_O = obj.ub_oriented - obj.lb_oriented
            
            prob_1 = axis_lengths_O[0] / np.sum(axis_lengths_O)
            prob_2 = axis_lengths_O[1] / np.sum(axis_lengths_O)
            num = np.random.rand()
            if num <= prob_1:
                hand_y_axis_ind = 0
            elif num <= prob_1 + prob_2:
                hand_y_axis_ind = 1
            else:
                hand_y_axis_ind = 2

            hand_y_axis_O = np.zeros(3)
            hand_y_axis_O[hand_y_axis_ind] = 1.0
            hand_y_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_y_axis_O

            # Add randomization to y-axis
            rand_sign = 1 if np.random.random() < 0.5 else -1
            _y_hat = rand_sign * hand_y_axis_W
            _y_hat = _y_hat / np.linalg.norm(_y_hat)
            y_hat = np.random.vonmises(mu=_y_hat, kappa=20)
            y_hat = y_hat / np.linalg.norm(y_hat)
            
            # Project onto plane perpendicular to x_hat and normalize
            y_hat = y_hat - (np.dot(y_hat, x_hat) * x_hat)
            y_hat = y_hat / np.linalg.norm(y_hat)

            # Complete right-handed frame
            z_hat = np.cross(x_hat, y_hat)

            if sampler.z_axis_fwd and z_hat[0] < 0:
                y_hat, z_hat = -y_hat, -z_hat

            # Form pose
            R_WPalm = np.stack((x_hat, y_hat, z_hat), axis=1)
            if isinstance(self.palm_offset, tuple):
                offset = np.random.uniform(*self.palm_offset)
            else:
                offset = self.palm_offset
            P_WPalm = contact_point_W + offset * normal_W

            # Check if the palm has negative z
            if P_WPalm[2] < 0:
                sucess = False
            else:
                sucess = True

        # Find the contacts opposite to the sampled contact
        vertices_W = (X_WO @ vertices.T).T
        normals_W = (obj.X_WO.rotation().matrix() @ normals.T).T
        opposite_contacts = find_opposite_contacts(
            contact_point_W, normal_W, vertices_W, normals_W, visualize=False,
            angle_tol=np.pi/6, min_dist=0.01, max_dist=0.08
        )
        sampler.support_contacts = opposite_contacts

        return RigidTransform(RotationMatrix(R_WPalm), P_WPalm)
    

class ActuationContactConstraint(CustomConstraint):
    """Implementation of constraint adding from FnHeuristicAlgrICSampler"""
    def __init__(self, actuation_contacts: list[tuple[np.ndarray, np.ndarray|None]], reverse_actuation=False):
        self.actuation_contacts = actuation_contacts
        self.reverse_actuation = reverse_actuation
        self.act_assignments = {}

    def add_additional_constraints(
        self,
        sampler: ConfigurableHeuristicICSampler,
        ik: InverseKinematics,
        X_WPalm_des: RigidTransform
    ) -> None:
        normal_tol_pos = 0.03
        fn_tol_rot = 1.3

        # 1. Set a default hand configuration guess
        q_hand = np.zeros(16)
        q_curr = sampler.model.plant.GetPositions(sampler.model.plant_context)
        q_curr[-16:] = q_hand
        sampler.model.plant.SetPositions(sampler.model.plant_context, q_curr)

        # 2. Compute width w based on object orientation and palm alignment
        obj = sampler.model.obj
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        R_OBB = X_OBB.rotation()
        R_WBB = X_WO.rotation() @ R_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        z_pc = X_WPalm_des.rotation().matrix()[:, 2]  # palm's z-axis
        z_alignment = np.argmax(np.abs(R_WBB.inverse() @ z_pc))
        w = axis_lengths_O[z_alignment]

        x_pc = X_WPalm_des.rotation().matrix()[:, 0]  # palm's x-axis
        x_alignment = np.argmax(np.abs(R_WBB.inverse() @ x_pc))
        if x_alignment == 2:
            f_ext = min(X_WPalm_des.translation()[-1], 0.08)
        else:
            f_ext = 0.08

        # 3. Compute normal fingertip target positions
        hand = sampler.model.hand
        if hand == "rh":
            p_if = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])   # index
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])    # middle
            p_rf = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])  # ring
            p_th = X_WPalm_des @ np.array([f_ext, 0.02, -w / 2])  # thumb
        else:
            p_if = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])
            p_rf = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])
            p_th = X_WPalm_des @ np.array([f_ext, -0.02, -w / 2])

        P_WFs = np.stack((p_if, p_mf, p_rf, p_th))

        # 4. Get the fingertip frames
        fingertip_bodies = [
            sampler.model.plant.GetBodyByName(name) for name in sampler.model.fingertip_names
        ]
        fingertip_frames = [b.body_frame() for b in fingertip_bodies]

        # Actuation contact assignment heuristic
        num_act = len(self.actuation_contacts)
        act_assignments = {}
        support_assignments = {}
        options = [i for i, name in enumerate(sampler.model.fingertip_names) if "palm" not in name]
        if num_act == 4:
            # All actuation contacts are present
            act_assignments = {i: self.actuation_contacts[i] for i in range(4)}
            options = []
        elif num_act > 0:
            # act_assignments[0] = self.actuation_contacts[0]
            # if num_act > 1:
            #     # Randomly assign the rest of the fingers to the rest of the actuation contacts (non repeating)
            for i in range(num_act):
                rnd_f = np.random.choice(options)
                options.remove(rnd_f)
                act_assignments[rnd_f] = self.actuation_contacts[i]
        
        # Support contacts assignment heuristics
        if hasattr(sampler, 'support_contacts') and sampler.support_contacts is not None:
            support_contacts = sampler.support_contacts
            contact_locs, contact_normals, contact_dists = support_contacts
            sampled_idx = np.random.choice(len(contact_locs))
            sampled_loc = contact_locs[sampled_idx]
            sampled_normal = contact_normals[sampled_idx]

            rnd_f = np.random.choice(options)
            options.remove(rnd_f)
            support_assignments[rnd_f] = (sampled_loc, -sampled_normal)
        
        # 50% chance of considering support contacts as actuation contacts
        if np.random.random() < 0.5:
            act_assignments.update(support_assignments)
        
        # Store the assignments inside the model
        sampler.model.act_assignments = act_assignments
        sampler.model.support_assignments = support_assignments
        fingertip_offsets = sampler.model.fingertip_offsets

        # Get object geometry ID
        # Retrieve the inspector from the query object.
        inspector = sampler.model.query_object.inspector()

        # Get all geometry IDs in the scene.
        all_geom_ids = inspector.GetAllGeometryIds()

        # Filter for object collision geometry IDs based on name.
        object_geom_ids = [gid for gid in all_geom_ids if "obj_collision" in inspector.GetName(gid)]

        # Similarly, get the fingertip geometry IDs.
        from collections import defaultdict
        fingertip_geom_ids = defaultdict(list)
        for gid in all_geom_ids:
            name = inspector.GetName(gid)
            for idx, fingertip_name in enumerate(sampler.model.fingertip_names):
                if fingertip_name + "_tip_collision" in name:
                    fingertip_geom_ids[idx].append(gid)

        # Add constraints
        from itertools import product
        for i, fingertip_name in enumerate(sampler.model.fingertip_names):
            # for obj_geom_id, fingertip_geom_id in product(object_geom_ids, fingertip_geom_ids[i]):
            #     ik.AddDistanceConstraint(
            #         (fingertip_geom_id, obj_geom_id),
            #         0.0,
            #         0.3
            #     )
            # for obj_geom_id in object_geom_ids:
            # ik.AddDistanceConstraint(
            #     (fingertip_geom_ids[i], obj_geom_id),
            #     0.0,
            #     0.05
            # )
            # Skip the palm
            if "palm" in fingertip_name:
                continue

            if i in act_assignments or i in support_assignments:
                # This finger has a actuation contact
                if i in act_assignments:
                    act_pos, act_dir = act_assignments[i]
                    if self.reverse_actuation:
                        act_dir = -act_dir
                
                    # Position constraint
                    ik.AddPositionConstraint(
                        fingertip_frames[i],
                        fingertip_offsets[i],
                        sampler.model.plant.world_frame(),
                        act_pos - 1e-3,
                        act_pos + 1e-3,
                    )
                    # Orientation constraint if act_dir is given
                    if act_dir is not None:
                        tol_angle_lower = 0
                        tol_angle_upper = 0.5  # Small tolerance in radians
                        # We want the x-axis of the finger frame to align with act_dir
                        finger_x_axis = np.array([1, 0, 0])  # x-axis vector in finger frame
                        
                        # Lower and upper bounds for the angle between vectors
                        ik.AddAngleBetweenVectorsConstraint(
                            fingertip_frames[i],      # Frame A (finger frame)
                            finger_x_axis,          # Vector in frame A (x-axis of finger)
                            sampler.model.plant.world_frame(),  # Frame B (world frame)
                            act_dir,                # Vector in frame B (actuation direction)
                            tol_angle_lower,            # Lower bound on angle between vectors
                            tol_angle_upper             # Upper bound on angle between vectors
                        )
                else:
                    sup_pos, sup_dir = support_assignments[i]
                    if self.reverse_actuation:
                        sup_dir = -sup_dir
                    tol_angle_lower = 0
                    tol_angle_upper = 0.5

                    # Position constraint
                    ik.AddPositionConstraint(
                        fingertip_frames[i],
                        fingertip_offsets[i],
                        sampler.model.plant.world_frame(),
                        sup_pos - 1e-4,
                        sup_pos + 1e-4,
                    )
                    # Orientation constraint if sup_dir is given
                    if sup_dir is not None:
                        # We want the z-axis of the finger frame to align with sup_dir
                        finger_z_axis = np.array([0, 0, 1])  # x-axis vector in finger frame
                        finger_x_axis = np.array([1, 0, 0])  # x-axis vector in finger frame
                        
                        # Lower and upper bounds for the angle between vectors
                        ik.AddAngleBetweenVectorsConstraint(
                            fingertip_frames[i],      # Frame A (finger frame)
                            finger_x_axis,          # Vector in frame A (x-axis of finger)
                            sampler.model.plant.world_frame(),  # Frame B (world frame)
                            sup_dir,                # Vector in frame B (actuation direction)
                            tol_angle_lower,            # Lower bound on angle between vectors
                            tol_angle_upper             # Upper bound on angle between vectors
                        )
            else:
                # Normal fingertip position constraint
                ik.AddPositionConstraint(
                    fingertip_frames[i],
                    fingertip_offsets[i],
                    sampler.model.plant.world_frame(),
                    P_WFs[i, :] - normal_tol_pos,
                    P_WFs[i, :] + normal_tol_pos,
                )
                pass

# Example usage:
def create_palm_predictor_sampler(
    model: RobotModel,
    contact_predictor,
    palm_offset: float = (0.005, 0.05),
    table: bool = False,
    z_axis_fwd: bool = False
) -> ConfigurableHeuristicICSampler:
    """Creates a sampler that uses contact prediction for palm sampling"""
    return ConfigurableHeuristicICSampler(
        model=model,
        custom_palm_sampler=ContactPalmSampler(contact_predictor, palm_offset),
        table=table,
        z_axis_fwd=z_axis_fwd
    )

def create_actuation_contact_sampler(
    model: RobotModel,
    actuation_contacts: list[tuple[np.ndarray, np.ndarray|None]],
    reverse_actuation: bool = False,
    table: bool = False,
    z_axis_fwd: bool = False
) -> ConfigurableHeuristicICSampler:
    """Creates a sampler that uses actuation contacts for constraints"""
    return ConfigurableHeuristicICSampler(
        model=model,
        custom_constraint=ActuationContactConstraint(actuation_contacts, reverse_actuation=reverse_actuation),
        table=table,
        z_axis_fwd=z_axis_fwd
    )

def create_hybrid_sampler(
    model: RobotModel,
    contact_predictor,
    actuation_contacts: list[tuple[np.ndarray, np.ndarray|None]],
    reverse_actuation: bool = False,
    palm_offset: float = (0.005, 0.05),
    table: bool = False,
    z_axis_fwd: bool = False
) -> ConfigurableHeuristicICSampler:
    """Creates a sampler that uses both contact prediction and actuation contacts"""
    return ConfigurableHeuristicICSampler(
        model=model,
        custom_palm_sampler=ContactPalmSampler(contact_predictor, palm_offset),
        custom_constraint=ActuationContactConstraint(actuation_contacts, reverse_actuation=reverse_actuation),
        table=table,
        z_axis_fwd=z_axis_fwd
    )