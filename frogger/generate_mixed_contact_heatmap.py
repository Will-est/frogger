import os
import pickle
import numpy as np
import open3d as o3d
from PIL import Image

from learning_based_heuristics import ContactDBHeuristic, ContactGenHeuristic

###############################################################################
# Function to combine predictions from both methods and save results
###############################################################################

def combine_contact_predictions(result_db, result_cg, 
                                weight_db=0.4, weight_cg=0.4, uniform_weight=0.2):
    """
    Args:
        result_db: tuple (heatmap_db, pts_db, normals_db) from ContactDB
        result_cg: tuple (heatmap_cg, pts_cg, normals_cg) from ContactGen
        weight_db: default weight for ContactDB predictions (0.4)
        weight_cg: default weight for ContactGen predictions (0.4)
        uniform_weight: base uniform weight (0.2)
        
    Returns:
        A dictionary with keys 'heatmap', 'pts', and 'normals'
        where the heatmap is computed as:
            candidate_score = (method_weight * method_heat) + (w_uniform_total / N_total)
        for candidates from each method.
        (If a method’s heatmap is entirely zero, its weight is added to the uniform weight.)
    """
    heatmap_db, pts_db, normals_db = result_db
    heatmap_cg, pts_cg, normals_cg = result_cg

    # Adjust weights if a method's heatmap is zero.
    if np.max(heatmap_db) == 0:
        weight_db = 0.0
        uniform_weight += 0.4
    if np.max(heatmap_cg) == 0:
        weight_cg = 0.0
        uniform_weight += 0.4

    # Concatenate candidates.
    pts_combined = np.concatenate([pts_db, pts_cg], axis=0)
    normals_combined = np.concatenate([normals_db, normals_cg], axis=0)
    N_total = pts_combined.shape[0]
    uniform_component = uniform_weight / N_total

    # Normalize heatmaps.
    # if np.max(heatmap_db) > 0:
    #     heatmap_db = heatmap_db / np.sum(heatmap_db)
    # if np.max(heatmap_cg) > 0:
    #     heatmap_cg = heatmap_cg / np.sum(heatmap_cg)

    # Combine heatmaps.
    combined_heatmap_db = (weight_db * heatmap_db) + uniform_component
    combined_heatmap_cg = (weight_cg * heatmap_cg) + uniform_component
    heatmap_combined = np.concatenate([combined_heatmap_db, combined_heatmap_cg], axis=0)

    return {
        'heatmap': heatmap_combined,
        'pts': pts_combined,
        'normals': normals_combined
    }

def process_and_save_mesh(mesh, mesh_name,
                          output_dir="output/mixed_heatmap/",
                          weight_db=0.4, weight_cg=0.4, uniform_weight=0.2,
                          visualize=False):
    """
    Processes a mesh using both ContactDB and ContactGen predictors,
    combines the results, and saves the result to a pickle file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictors.
    contactdb_model = ContactDBHeuristic(precompute=True)
    contactgen_model = ContactGenHeuristic(precompute=True)
    
    # Load and query.
    print(f"Processing {mesh_name} with ContactDB...")
    contactdb_model.load_object(mesh)
    result_db = contactdb_model.query(visualize=False)
    
    print(f"Processing {mesh_name} with ContactGen...")
    contactgen_model.load_object(mesh)
    result_cg = contactgen_model.query(visualize=False)
    
    # Combine results.
    combined_result = combine_contact_predictions(result_db, result_cg,
                                                    weight_db=weight_db,
                                                    weight_cg=weight_cg,
                                                    uniform_weight=uniform_weight)
    
    # Visualize if requested.
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.point_size = 2.0

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(combined_result['pts'])

        max_value = np.max(combined_result['heatmap'])
        colors = np.zeros((combined_result['pts'].shape[0], 3))
        colors[:, 0] = combined_result['heatmap'] / max_value
        colors[:, 2] = 1 - combined_result['heatmap'] / max_value
        pc.colors = o3d.utility.Vector3dVector(colors)
        pc.normals = o3d.utility.Vector3dVector(combined_result['normals'])
        
        vis.add_geometry(pc)
        vis.run()
        image = vis.capture_screen_float_buffer(True)
        image_np = np.asarray(image)
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(os.path.join(output_dir, f"{mesh_name}.png"))
        vis.destroy_window()
    
    # Save to file.
    file_path = os.path.join(output_dir, f"{mesh_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(combined_result, f)
    print(f"Saved combined heatmap to {file_path}")

if __name__ == "__main__":
    from dexfun.mesh.load_mesh import load_mesh
    mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"
    for mesh_name in os.listdir(mesh_dir):
        # load_mesh should return a trimesh (or similar) mesh.
        mesh, _ = load_mesh(mesh_dir, mesh_name, mesh_format="obj")
        process_and_save_mesh(mesh, mesh_name, visualize=True)
