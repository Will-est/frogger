import numpy as np
import trimesh
import json
import os
import glob
from pydrake.math import RigidTransform, RotationMatrix
from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.solvers import FroggerConfig
from frogger.custom_robot_model import FunctionalRobotModel, custom_fingertip_height_constraints
from frogger.custom_solver import FunctionalFrogger
from frogger.utils import timeout
from frogger.custom_sampling import create_hybrid_sampler
from frogger.learning_based_heuristics import ContactMixedHeuristic
from dexfun.mesh.load_mesh import load_mesh
import gc

# Define the tolerance dictionary.
# Note: tol level 1 is the strictest and 4 is the loosest.
tols = {
    1: {"tol_surf": 1e-3, "tol_joint": 1e-2, "tol_col": 1e-3, "tol_fclosure": 1e-2, "timeout": 240.0, "normal_contact_weight": 5e-1},
    2: {"tol_surf": 2e-3, "tol_joint": 1e-2, "tol_col": 2e-3, "tol_fclosure": 2e-1, "timeout": 300.0, "normal_contact_weight": 3e-1},
    3: {"tol_surf": 4e-3, "tol_joint": 1e-2, "tol_col": 4e-3, "tol_fclosure": 2e-1, "timeout": 400.0, "normal_contact_weight": 2e-1},
    4: {"tol_surf": 8e-3, "tol_joint": 1e-2, "tol_col": 8e-3, "tol_fclosure": 2e-1, "timeout": 600.0, "normal_contact_weight": 1e-1},
}


def attempt_grasp_for_mesh(mesh_dir, actuation_dir, output_dir, mesh_name, tol_level, optimize=True):
    """
    Loads the mesh and its associated data, builds the model and sampler,
    then attempts to generate a grasp using the given tolerance level.
    Saves grasp files (configuration, optionally optimized configuration, and metadata)
    into the mesh-specific output folder.
    Returns True if a grasp is successfully generated.
    """    
    # Create output directory for this mesh
    mesh_output_dir = os.path.join(output_dir, mesh_name)
    os.makedirs(mesh_output_dir, exist_ok=True)

    # Load mesh and compute offset (to put the base at z=0)
    try:
        mesh, _ = load_mesh(mesh_dir, mesh_name, mesh_format="obj")
    except Exception as e:
        raise Exception(f"Error loading mesh {mesh_name}: {e}")
    
    bounds = mesh.bounds
    lb_O = bounds[0, :]
    offset = np.array([0.0, 0.0, -lb_O[-1]])
    X_WO = RigidTransform(RotationMatrix(), offset)
    
    # Create object config
    try:
        obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=mesh_name, clean=False).create()
    except Exception as e:
        raise Exception(f"Error creating object config for {mesh_name}: {e}")

    # Load actuation contacts and apply offset
    actuation_filepath = os.path.join(actuation_dir, f"{mesh_name}_actuation.json")
    actuation_contacts = []
    try:
        if os.path.exists(actuation_filepath):
            with open(actuation_filepath, "r") as f:
                actuation_info = json.load(f)
                actuation_contacts = actuation_info["actuation_contacts"]
                actuation_contacts = [
                    (np.array(contact["pos"]) + offset, np.array(contact["dir"]))
                    for contact in actuation_contacts
                ]
    except Exception as e:
        print(f"Error loading actuation contacts for {mesh_name}: {e}")
    
    if len(actuation_contacts) == 0:
        raise Exception(f"No actuation contacts found for {mesh_name}")

    # Create the model configuration and the functional robot model.
    model_cfg = AlgrModelConfig(
        obj=obj,
        ns=4,
        mu=0.8,
        d_min=0.001,
        d_pen=0.005,
        l_bar_cutoff=0.3,
        hand="rh",
        # custom_compute_g=custom_fingertip_height_constraints,
        # n_g_extra=4,
    )
    model = FunctionalRobotModel(model_cfg)
    model.normal_contact_weight = tols[tol_level]["normal_contact_weight"]

    # Set up the sampler using a contact predictor.
    contact_predictor = ContactMixedHeuristic(heatmap_dir="output/mixed_heatmap")
    contact_predictor.load_object(mesh_name=mesh_name)
    sampler = create_hybrid_sampler(
        model, 
        contact_predictor, 
        actuation_contacts, 
        palm_offset=(0.005, 0.02)
    )

    # Create a new solver instance.
    frogger = FunctionalFrogger(
        cfg=FroggerConfig(
            model=model,
            sampler=sampler,
            tol_surf=tols[tol_level]["tol_surf"],
            tol_joint=tols[tol_level]["tol_joint"],
            tol_col=tols[tol_level]["tol_col"],
            tol_fclosure=tols[tol_level]["tol_fclosure"],
            xtol_rel=1e-5,
            xtol_abs=1e-5,
            maxeval=500,
        ),
        actuation_contacts=actuation_contacts,
    )
    
    # Run the grasp generation with a timeout.
    # timeout_func = timeout(tols[tol_level]["timeout"])
    try:
        q_star, q0 = frogger.generate_grasp(
            optimize=optimize,
            timeout_seconds=tols[tol_level]["timeout"],
            check_constraints=False,
            tol_pos=0.04,
            tol_ang=0.2
        )
    except TimeoutError:
        print(f"  {mesh_name} tol {tol_level} attempt error. Timeout.")
        raise TimeoutError
    except Exception as e:
        raise e
        
    
    if q0 is None:
        raise Exception("generate_grasp returned no result.")
    
    # Determine a new grasp index (based on the number of metadata files already saved)
    existing_metadata = glob.glob(os.path.join(mesh_output_dir, "grasp_*_metadata.json"))
    grasp_id = len(existing_metadata) + 1

    # Save the grasp configuration.
    grasp_path = os.path.join(mesh_output_dir, f"grasp_{grasp_id}.txt")
    np.savetxt(grasp_path, q0)
    
    # Save the optimized grasp (if available and if optimize==True).
    if optimize and q_star is not None:
        optimized_path = os.path.join(mesh_output_dir, f"grasp_{grasp_id}_optimized.txt")
        np.savetxt(optimized_path, q_star)
    
    # Get actuation finger indices from the model.
    act_correspondence = model.contact_correspondence
    act_finger_idx = [i for i, j in enumerate(act_correspondence) if j != -1]
    
    # Save metadata.
    metadata = {
        "mesh_name": mesh_name,
        "tolerance_level": tol_level,
        "tolerances": tols[tol_level],
        "optimize": optimize,
        "success": True,
        "actuation_finger_indices": act_finger_idx,
    }
    metadata_path = os.path.join(mesh_output_dir, f"grasp_{grasp_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved grasp {grasp_id} for {mesh_name}")

    del frogger, model, sampler, obj

    return grasp_id


def main():
    # Define directories.
    mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
    actuation_dir = "/home/bowenj/Projects/DexFun/output/actuation_contacts/mesh_raw_ahg"
    output_dir = "/home/bowenj/Projects/DexFun/output/grasps_rh_v3/mesh_raw_ahg"
    os.makedirs(output_dir, exist_ok=True)
    
    import sys
    mesh_name = sys.argv[1]
    mesh_names = [mesh_name]
    # Get all mesh names from the mesh directory.
    # mesh_names = list(os.listdir(mesh_dir))
    # mesh_names = [
    #     # "funky_clear_spray_bottle",
    #     # "clear_spray_bottle_single",
    #     # "zip_tie_gun_single",
    #     # "pen_single",
    #     # "black_spray_bottle_single",
    #     # "black_febreeze_small_single",
    #     # "clamp_single",
    #     # "scissors_single",
    #     # "hot_glue_gun",
    #     # "pliers_single",
    #     # "marker_single",
    #     # "syrup_pourer_single",
    #     # "black_febreeze_tall",
    #     # "knife_single",
    #     # "pump_spray_single",
    #     # "black_febreeze_tall_single",
    #     # "staple_remover_single",
    #     # "zip_tie_tool_single"
    # ]
    # mesh_names = ["hot_glue_gun"]
    
    # Save run metadata.
    run_metadata = {
        "mesh_dir": mesh_dir,
        "actuation_dir": actuation_dir,
        "output_dir": output_dir,
        "mesh_count": len(mesh_names),
        "mesh_names": mesh_names,
        "grasps_per_mesh": 5,
        "timestamp": np.datetime_as_string(np.datetime64('now'))
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    # Keep minimal status info in memory.
    mesh_status = {}
    for mesh_name in mesh_names:
        mesh_status[mesh_name] = {"successful_grasps": 0}
    
    target_grasps = run_metadata["grasps_per_mesh"]

    # WIDTH-FIRST PASS: For each tolerance level (from strictest to loosest), run through all objects.
    for tol_level in [3,4]:
        print(f"\nStarting pass with tolerance level {tol_level}")
        for mesh_name in mesh_names:
            mesh_output_dir = os.path.join(output_dir, mesh_name)
            existing_metadata = glob.glob(os.path.join(mesh_output_dir, "grasp_*_metadata.json"))
            mesh_status[mesh_name]["successful_grasps"] = len(existing_metadata)
            if mesh_status[mesh_name]["successful_grasps"] >= target_grasps:
                continue
            failures = 0
            while mesh_status[mesh_name]["successful_grasps"] < target_grasps and failures < 3:
                try:
                    success_grasps = mesh_status[mesh_name]["successful_grasps"]
                    print(f"Attempting grasp for {mesh_name} at tol level {tol_level}. Progress: {success_grasps}/{target_grasps}. Current failres: {failures}")
                    grasp_id = attempt_grasp_for_mesh(mesh_dir, actuation_dir, output_dir, mesh_name, tol_level, optimize=True)
                    mesh_status[mesh_name]["successful_grasps"] = grasp_id
                except TimeoutError:
                    print(f"  {mesh_name} tol {tol_level} attempt error. Timeout.")
                    failures += 1
                except Exception as e:
                    print(f"  {mesh_name} tol {tol_level} attempt error: {e}")
                    print(f"  Traceback: {e.__traceback__}")
                    raise e
            print("Current progress:")
            print(f"  {mesh_name}: {mesh_status[mesh_name]['successful_grasps']}/{target_grasps}")
            # gc.collect()
        
        # Report overall progress.
        total_grasps = sum([mesh_status[m]["successful_grasps"] for m in mesh_status])
        expected_total = len(mesh_names) * target_grasps
        print(f"Completed pass with tolerance level {tol_level}.")
        print(f"Total successful grasps: {total_grasps}/{expected_total}")
    
    # FINAL PASS: For any mesh that still hasn't reached target grasps, run with optimize=False.
    print("\nStarting final pass (optimize=False) for remaining meshes.")
    for mesh_name in mesh_names:
        if mesh_status[mesh_name]["successful_grasps"] >= target_grasps:
            continue
        failures = 0
        while mesh_status[mesh_name]["successful_grasps"] < target_grasps and failures < 2:
            try:
                # Use the loosest tolerance level (4) for the final pass.
                print(f"Attempting grasp for {mesh_name} at tol level {tol_level}. Current failres: {failures}")
                grasp_id = attempt_grasp_for_mesh(mesh_dir, actuation_dir, output_dir, mesh_name, 4, optimize=False)
                mesh_status[mesh_name]["successful_grasps"] = grasp_id
            except TimeoutError:
                print(f"  {mesh_name} tol {tol_level} attempt error. Timeout.")
                failures += 1
            except Exception as e:
                print(f"  {mesh_name} tol {tol_level} attempt error: {e}")
                print(f"  Traceback: {e.__traceback__}")
                raise e
    
    # Save summary of results.
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(mesh_status, f, indent=2)
    
    total_grasps = sum([mesh_status[m]["successful_grasps"] for m in mesh_status])
    expected_total = len(mesh_names) * target_grasps
    print(f"\nCompleted processing {len(mesh_names)} meshes.")
    print(f"Total successful grasps: {total_grasps}/{expected_total}")


if __name__ == "__main__":
    main()
