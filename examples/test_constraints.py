import numpy as np
import os
import trimesh
import open3d as o3d
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.geometry import Sphere, Rgba

from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.robots.custom_robots import LeapModelConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.solvers import FroggerConfig, Frogger
# Import our custom classes
from frogger.custom_robot_model import FunctionalRobotModel  # Updated import
from frogger.custom_solver import FunctionalFrogger         # Updated import
from frogger.utils import timeout

from frogger.custom_sampling import create_actuation_contact_sampler
from frogger.learning_based_heuristics import ContactDBHeuristic, ContactGenHeuristic

# -------------------- Main Code --------------------

from dexfun.mesh.load_mesh import load_mesh, trimesh_to_o3d
import json

def load_mesh_and_contacts(mesh_name, mesh_dir, actuation_dir):
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
    functional_contacts = None
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

    return obj, actuation_contacts, functional_contacts

def create_frogger(obj, create_frogger=False):
     # Create the configuration
    # model_cfg = LeapModelConfig(
    model_cfg = AlgrModelConfig(
        obj=obj,
        ns=4,
        mu=0.9,
        d_min=0.001,
        d_pen=0.005,
        l_bar_cutoff=0.3,
        hand="rh",
    )

    # Compute the unconstrained pose
    # model = model_cfg.create()
    model = FunctionalRobotModel(model_cfg)  # # Create our functional robot model instead of regular model

    if create_frogger:
        # Sampler selection remains the same
        if len(actuation_contacts) > 0:
            sampler = create_actuation_contact_sampler(model, actuation_contacts)
            # contact_predictor = ContactDBHeuristic()
            # sampler = ContactHeuristicAlgrICSampler(model=model, functional_contacts=functional_contacts, contact_predictor=contact_predictor)
        else:
            sampler = HeuristicAlgrICSampler(model)

        # Create our functional solver
        frogger = FunctionalFrogger(  # Changed from FnFrogger
            cfg=FroggerConfig(
                model=model,
                sampler=sampler,
                tol_surf=1e-2,      # 1e-3
                tol_joint=1e-1,     # 1e-2
                tol_col=1e-2,       # 1e-3
                tol_fclosure=1e-3,  # relaxed
                xtol_rel=1e-4,      # relaxed
                xtol_abs=1e-4,      # relaxed
                maxeval=1000,
            ),
            actuation_contacts=actuation_contacts,
        )
        return model, sampler, frogger
    else:
        return model, None, None

def test_constraints(grasp_path, model: FunctionalRobotModel, sampler=None, frogger=None):
    # Load a grasp pose
    
    # grasp_path = f"{grasp_dir}/{mesh_name}/grasp_1_optimized.txt"
    # grasp_path = f"{grasp_dir}/{mesh_name}/grasp_1.txt"
    # grasp_path = "/home/bowenj/Projects/DexFun/output/grasps_rh/mesh_raw_ahg/black_spray_bottle_single/grasp_1_optimized.txt"
    q = np.loadtxt(grasp_path)
    q *= 0.
    q[3] = 1.0

    # Generate grasp
    if frogger is not None:
        frogger.check_constraints(q)

    # if hasattr(frogger.sampler, "add_visualization"):
    #     frogger.sampler.add_visualization(model)

    # # Visualize configurations
    model.viz_config(q)


if __name__ == "__main__":
    # Initialize the mesh and contacts
    mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
    actuation_dir = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"
    mesh_name = "black_spray_bottle_single"
    # mesh_name = "black_febreeze_small_single"
    obj, actuation_contacts, functional_contacts = load_mesh_and_contacts(mesh_name, mesh_dir, actuation_dir)
    # model, sampler, frogger = create_frogger(obj)
    model, sampler, frogger = create_frogger(obj, create_frogger=False)

    # Test constraints
    grasp_dir = "/home/bowenj/Projects/DexFun/output/grasps_rh_v2/mesh_raw_ahg"
    grasp_path = f"{grasp_dir}/{mesh_name}/grasp_1_optimized.txt"
    print(f"Testing grasp: {grasp_path}")
    test_constraints(grasp_path, model, sampler, frogger)

    # Batch test
    # grasp_dir = f"/home/bowenj/Projects/DexFun/output/grasps_rh_v2/mesh_raw_ahg/{mesh_name}"
    # for grasp_name in os.listdir(grasp_dir):
    #     if grasp_name.endswith(".txt"):
    #         print(f"Testing grasp: {grasp_name}")
    #         grasp_path = f"{grasp_dir}/{grasp_name}"
    #         test_constraints(model, sampler, frogger, grasp_path)



