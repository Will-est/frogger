import numpy as np
import trimesh
import json
import os
import glob
from pydrake.math import RigidTransform, RotationMatrix
from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.solvers import FroggerConfig
from frogger.custom_robot_model import FunctionalRobotModel
from frogger.custom_solver import FunctionalFrogger
from frogger.utils import timeout
from frogger.custom_sampling import create_hybrid_sampler
from frogger.learning_based_heuristics import ContactMixedHeuristic
from dexfun.mesh.load_mesh import load_mesh

def predict_grasps_for_mesh(mesh_dir, actuation_dir, output_dir, mesh_name, num_grasps=5):
    print(f"Processing mesh: {mesh_name}")
    
    # Create output directory for this mesh
    mesh_output_dir = os.path.join(output_dir, mesh_name)
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    # Load mesh
    try:
        mesh, _ = load_mesh(mesh_dir, mesh_name, mesh_format="obj")
    except Exception as e:
        print(f"Error loading mesh {mesh_name}: {e}")
        return
    
    # Process mesh
    bounds = mesh.bounds
    lb_O = bounds[0, :]
    ub_O = bounds[1, :]
    offset = np.array([0.0, 0.0, -lb_O[-1]])  # Offset the mesh to the origin
    X_WO = RigidTransform(RotationMatrix(), offset)
    
    try:
        obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=mesh_name, clean=False).create()
    except Exception as e:
        print(f"Error creating object config for {mesh_name}: {e}")
        return
    
    # Load actuation contacts
    actuation_filepath = f"{actuation_dir}/{mesh_name}_actuation.json"
    actuation_contacts = []
    
    try:
        if os.path.exists(actuation_filepath):
            with open(actuation_filepath, "r") as f:
                actuation_info = json.load(f)
                actuation_contacts = actuation_info["actuation_contacts"]
                # Reformat the actuation contacts
                actuation_contacts = [
                    (np.array(contact["pos"]) + offset, np.array(contact["dir"])) 
                    for contact in actuation_contacts
                ]
    except Exception as e:
        print(f"Error loading actuation contacts for {mesh_name}: {e}")
        # Continue with empty actuation contacts
    
    # Create the model configuration
    model_cfg = AlgrModelConfig(
        obj=obj,
        ns=4,
        mu=0.9,
        d_min=0.001,
        d_pen=0.005,
        l_bar_cutoff=0.3,
        hand="lh",
    )
    
    # Create functional robot model
    model = FunctionalRobotModel(model_cfg)
    
    # Setup sampler
    assert len(actuation_contacts) > 0, "No actuation contacts found"
    contact_predictor = ContactMixedHeuristic(heatmap_dir="output/mixed_heatmap")
    contact_predictor.load_object(mesh_name=mesh_name)
    sampler = create_hybrid_sampler(
        model, 
        contact_predictor, 
        actuation_contacts, 
        palm_offset=(0.005, 0.02)
    )
    
    # Generate and save grasps
    successful_grasps = 0
    attempt = 0
    max_attempts = num_grasps * 2  # Allow for some failures
    
    # Tolerances
    tols = {
        4: {"tol_surf": 8e-3, "tol_joint": 1e-2, "tol_col": 8e-3, "tol_fclosure": 2e-1, "timeout": 600.0},
        3: {"tol_surf": 4e-3, "tol_joint": 1e-2, "tol_col": 4e-3, "tol_fclosure": 2e-1, "timeout": 360.0},
        2: {"tol_surf": 2e-3, "tol_joint": 1e-2, "tol_col": 2e-3, "tol_fclosure": 2e-1, "timeout": 240.0},
        1: {"tol_surf": 1e-3, "tol_joint": 1e-2, "tol_col": 1e-3, "tol_fclosure": 1e-2, "timeout": 120.0},
    }
    tol_lvl = 1
    while successful_grasps < num_grasps and attempt < max_attempts:
        try:
            # Create solver
            frogger = FunctionalFrogger(
                cfg=FroggerConfig(
                    model=model,
                    sampler=sampler,
                    tol_surf=tols[tol_lvl]["tol_surf"],
                    tol_joint=tols[tol_lvl]["tol_joint"],
                    tol_col=tols[tol_lvl]["tol_col"],
                    tol_fclosure=tols[tol_lvl]["tol_fclosure"],
                    # tol_joint=1e-2,
                    # tol_col=5e-3,
                    # tol_fclosure=2e-1,
                    xtol_rel=1e-5,
                    xtol_abs=1e-5,
                    maxeval=500,
                ),
                actuation_contacts=actuation_contacts,
            )
            print(f"  Generating grasp {successful_grasps+1}/{num_grasps} (attempt {attempt+1})")
            q_star, q0 = timeout(tols[tol_lvl]["timeout"])(frogger.generate_grasp)(
                optimize=True, 
                check_constraints=False, 
                tol_pos=0.04, 
                tol_ang=0.2
            )
            act_correspondence = model.contact_correspondence
            act_finger_idx = [i for i, j in enumerate(act_correspondence) if j != -1]
            
            if q0 is not None:
                # Save grasp configuration
                grasp_path = os.path.join(mesh_output_dir, f"grasp_{successful_grasps+1}.txt")
                np.savetxt(grasp_path, q0)
                
                # Save optimized grasp if available
                if q_star is not None:
                    optimized_path = os.path.join(mesh_output_dir, f"grasp_{successful_grasps+1}_optimized.txt")
                    np.savetxt(optimized_path, q_star)
                
                # Save metadata
                metadata = {
                    "mesh_name": mesh_name,
                    "tolerance_level": tol_lvl,
                    "tolerances": tols[tol_lvl],
                    "success": True,
                    "actuation_finger_indices": act_finger_idx,
                }
                metadata_path = os.path.join(mesh_output_dir, f"grasp_{successful_grasps+1}_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                successful_grasps += 1
                print(f"  Saved grasp {successful_grasps}/{num_grasps}")
        except Exception as e:
            if tol_lvl < 4:
                tol_lvl += 1
            print(f"  Error generating grasp: {e}")
        
        attempt += 1
    
    print(f"Completed {mesh_name}: {successful_grasps}/{num_grasps} grasps generated")
    return successful_grasps


def main():
    # Define directories
    mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
    actuation_dir = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"
    output_dir = "/home/bowenj/Projects/DexFun/output/grasps/mesh_raw_ahg"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mesh names from the directory
    mesh_names = list(os.listdir(mesh_dir))
    
    # Save metadata about the run
    metadata = {
        "mesh_dir": mesh_dir,
        "actuation_dir": actuation_dir,
        "output_dir": output_dir,
        "mesh_count": len(mesh_names),
        "mesh_names": mesh_names,
        "grasps_per_mesh": 10,
        "timestamp": np.datetime_as_string(np.datetime64('now'))
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Process each mesh
    results = {}
    for mesh_name in mesh_names:
        try:
            successful_grasps = predict_grasps_for_mesh(
                mesh_dir=mesh_dir,
                actuation_dir=actuation_dir,
                output_dir=output_dir,
                mesh_name=mesh_name
            )
            results[mesh_name] = successful_grasps
        except Exception as e:
            print(f"Error processing mesh {mesh_name}: {e}")
            results[mesh_name] = 0
    
    # Save summary of results
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed processing all {len(mesh_names)} meshes.")
    print(f"Total successful grasps: {sum(results.values())}/{len(mesh_names)*10}")


if __name__ == "__main__":
    main()