import numpy as np
import trimesh
import json
import open3d as o3d
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.geometry import Sphere, Rgba

from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.robots.custom_robots import LeapModelConfig
from frogger.sampling import HeuristicAlgrICSampler
# from frogger.custom_sampling import FnHeuristicAlgrICSampler
from frogger.solvers import FroggerConfig, Frogger
# Import our custom classes
from frogger.custom_robot_model import FunctionalRobotModel, custom_fingertip_height_constraints
from frogger.custom_solver import FunctionalFrogger         # Updated import
from frogger.utils import timeout

# from frogger.custom_sampling import ContactHeuristicAlgrICSampler, PalmHeuristicAlgrICSampler
from frogger.custom_sampling import create_hybrid_sampler, create_palm_predictor_sampler, create_actuation_contact_sampler
from frogger.learning_based_heuristics import ContactDBHeuristic, ContactGenHeuristic, ContactMixedHeuristic
from frogger.utils import add_marker

# -------------------- Main Code --------------------

from dexfun.mesh.load_mesh import load_mesh, trimesh_to_o3d

mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
actuation_dir = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"

# mesh_name = "hot_glue_gun"
mesh_name = "black_spray_bottle_single"
mesh, _ = load_mesh(mesh_dir, mesh_name, mesh_format="obj")

# Load and process mesh - this part remains the same
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
offset = np.array([0.0, 0.0, -lb_O[-1]])    # Offset the mesh to the origin
X_WO = RigidTransform(
    RotationMatrix(),
    offset,
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=mesh_name, clean=False).create()

# Load actuation contacts
actuation_filepath = f"{actuation_dir}/{mesh_name}_actuation.json"
with open(actuation_filepath, "r") as f:
    actuation_info = json.load(f)
    actuation_contacts = actuation_info["actuation_contacts"]

    # Reformat the actuation contacts
    actuation_contacts = [
        (np.array(contact["pos"]) + offset , np.array(contact["dir"])) 
        for contact in actuation_contacts
    ]

# Load functional contacts
# functional_dir = "/home/bowenj/Projects/DexFun/output/functional_contacts/mesh_raw_ahg"
# functional_filepath = f"{functional_dir}/{mesh_name}.json"
# with open(functional_filepath, "r") as f:
#     functional_info = json.load(f)
#     functional_contacts = functional_info["functional_contacts"]

#     # Reformat the functional contacts
#     functional_contacts = [
#         (np.array(contact["pos"]) + offset , np.array(contact["dir"])) 
#         for contact in functional_contacts
#     ]

# Convert mesh and get functional contacts - this remains the same
processed_mesh = trimesh_to_o3d(mesh)

# support_contacts = [
#     (np.array([-0.01386208,  0.04284091,  0.07107334]), np.array([ 0.98006238,  0.05186685, -0.19180085])),
#     (np.array([0.02397209, 0.03807753, 0.06341889]), np.array([-0.97386469,  0.22416271, -0.03658755])),
#     ]
# # support_contacts = [
# #     (np.array([-0.01386208,  0.04284091,  0.07107334]), None),
# #     (np.array([0.02397209, 0.03807753, 0.06341889]), None)
# #     ]
# actuation_contacts = [
#     (np.array([ 0.00308108, -0.02312746,  0.07774291]), np.array([ 0.10211614,  0.75565293, -0.64696287])),
#     # *support_contacts,
#     ]

# Create the configuration
model_cfg = AlgrModelConfig(
# model_cfg = LeapModelConfig(
    obj=obj,
    ns=4,
    mu=0.9,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
    custom_compute_g=custom_fingertip_height_constraints,
    n_g_extra=4,
    # palm_contact=True,
)

reverse_actuation = False
if isinstance(model_cfg, LeapModelConfig):
    reverse_actuation = True

# Compute the unconstrained pose
# model = model_cfg.create()
model = FunctionalRobotModel(model_cfg)  # # Create our functional robot model instead of regular model

# Sampler selection remains the same
if len(actuation_contacts) > 0:
    # sampler = FnHeuristicAlgrICSampler(model, actuation_contacts)
    # contact_predictor = ContactDBHeuristic()
    # contact_predictor = ContactGenHeuristic()
    contact_predictor = ContactMixedHeuristic(heatmap_dir="output/mixed_heatmap")
    contact_predictor.load_object(mesh_name=mesh_name)
    sampler = create_hybrid_sampler(model, contact_predictor, actuation_contacts, reverse_actuation=reverse_actuation,
                                    palm_offset=(0.005, 0.02))
    # sampler = create_palm_predictor_sampler(model, contact_predictor, palm_offset=(0.005, 0.02))
    # sampler = create_actuation_contact_sampler(model, actuation_contacts, reverse_actuation=reverse_actuation)
    # sampler = PalmHeuristicAlgrICSampler(model, actuation_contacts, contact_predictor, palm_offset=0.05)
    # sampler = ContactHeuristicAlgrICSampler(model=model, actuation_contacts=actuation_contacts, contact_predictor=contact_predictor)
else:
    sampler = HeuristicAlgrICSampler(model)

# Create our functional solver
frogger = FunctionalFrogger(  # Changed from FnFrogger
    cfg=FroggerConfig(
        model=model,
        sampler=sampler,
        tol_surf=5e-3,      # 1e-3
        tol_joint=1e-2,     # 1e-2
        tol_col=7e-3,       # 1e-3
        tol_fclosure=2e-1,  # relaxed
        xtol_rel=1e-5,      # relaxed 1e-4
        xtol_abs=1e-5,      # relaxed 1e-4
        maxeval=500,       # 1000
    ),
    actuation_contacts=actuation_contacts,
)

# Generate grasp
print("Model compiled! Generating grasp...")
# q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=False, check_constraints=False, tol_pos=0.15, tol_ang=0.5)
q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=True, check_constraints=False, tol_pos=0.05, tol_ang=0.2)
print("Grasp generated!")

# fingertip_poses = model.compute_fingertip_poses()
# fingertip_poses = model.compute_contact_poses(q=q0)

# # Visualize the fingertips
# for i, pose in enumerate(fingertip_poses):
#     pos, _ = pose
#     # sphere = Sphere(0.0025)
#     model.meshcat.SetObject(
#         path=f"fingertip_{i}",
#         shape=Sphere(0.01),
#         rgba=Rgba(1.0, 0.0, 0.0, 1.0),
#     )
#     model.meshcat.SetTransform(
#         path=f"fingertip_{i}",
#         X_ParentPath=RigidTransform(pos)
#     )
# add_marker(model, sampler.X_WPalm_des.translation(), color=[0,1,0,1], radius=0.01, name="palm")
# model.set_q(q_star)
if hasattr(frogger.sampler, "add_visualization"):
    frogger.sampler.add_visualization(model)

# Visualize configurations
model.viz_config(q0)
if q_star is not None:
    model.viz_config(q_star)

# Generate grasps
# from copy import deepcopy
# results = []
# for i in range(10):
#     print("Model compiled! Generating grasp...")
#     q_star, q0 = timeout(1000.0)(frogger.generate_grasp)(optimize=False, check_constraints=False, tol_pos=0.01, tol_ang=0.1)
#     print("Grasp generated!")

#     # model_copy = deepcopy(model)
#     if hasattr(frogger.sampler, "add_visualization"):
#         frogger.sampler.add_visualization(model)

#     # results.append((model, q0))
#     model.viz_config(q0)

    


