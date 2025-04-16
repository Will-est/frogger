import os
import sys
import json
import torch
import numpy as np
import open3d
import trimesh
import pickle
from tqdm import tqdm

# Assume these are already defined in your codebase.
from contactdb.models.voxnet import DiverseVoxNet as VoxNet
from contactgen.model import ContactGenModel
from contactgen.utils.cfg_parser import Config

###############################################################################
# Base and existing heuristic classes (as defined previously)
###############################################################################

class ContactHeuristic:
    def __init__(self, precompute=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precompute = precompute
        self.current_object = None
        self.cache = None

    def query(self, visualize=False):
        """
        Returns:
            heatmap: np.ndarray of contact scores
            pts: np.ndarray of point candidates
            normals: np.ndarray of normals
        """
        if self.current_object is None:
            raise ValueError("No object loaded. Call load_object first.")
        if self.precompute:
            if self.cache is None:
                raise ValueError("No precomputed heatmaps available. Call load_object first.")
            heatmap, pts, normals = self._get_cached()
        else:
            heatmap, pts, normals = self._predict()

        if visualize:
            self._visualize_prediction(pts, heatmap)
        return heatmap, pts, normals

    def _visualize_prediction(self, pts, heatmap):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pts)
        heatmap_max = np.max(heatmap)
        colors = np.zeros((pts.shape[0], 3))
        colors[:, 0] = heatmap / heatmap_max
        colors[:, 2] = 1 - heatmap / heatmap_max
        pc.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pc])

    def _predict(self):
        raise NotImplementedError("Subclasses must implement _predict method.")

    def _get_cached(self):
        raise NotImplementedError("Subclasses must implement _get_cached method.")

###############################################################################
# ContactDBHeuristic and ContactGenHeuristic remain as before.
###############################################################################

class ContactDBHeuristic(ContactHeuristic):
    def __init__(self,
                 checkpoint_path="/home/bowenj/Projects/DexFun/third_parties/contactdb_prediction/data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss=0.01107167.pth",
                 grid_size=64,
                 precompute=True):
        super().__init__(precompute=precompute)
        self.grid_size = grid_size
        # Temporary module hack.
        original_module = sys.modules.get('models', None)
        sys.modules['models'] = sys.modules.get('contactdb.models')
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            from contactdb.models.voxnet import DiverseVoxNet as VoxNet
            self.model = VoxNet(n_ensemble=checkpoint.n_ensemble, droprate=0.1)
            self.model.voxnet.load_state_dict(checkpoint.voxnet.state_dict())
        finally:
            if original_module is None:
                del sys.modules['models']
            else:
                sys.modules['models'] = original_module

        self.model.to(self.device)
        self.model.eval()
        self.vertices = None
        self.normals = None

    def load_object(self, mesh):
        self.vertices = np.array(mesh.vertices)
        self.normals = np.array(mesh.vertex_normals)

        min_bound = self.vertices.min(axis=0)
        max_bound = self.vertices.max(axis=0)
        voxel_size = (max_bound - min_bound).max() / (self.grid_size - 1)

        # Voxel coordinates.
        voxel_coords = ((self.vertices - min_bound) / voxel_size).astype(int)
        x, y, z = voxel_coords.T
        pts = self.vertices.T

        # Center and scale.
        offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
        pts -= offset
        scale = (pts.max(1) - pts.min(1)).max() / 2
        pts /= scale
        pts = np.vstack((np.ones(pts.shape[1]), pts, scale * np.ones(pts.shape[1])))

        # Center in grid.
        offset_x = (self.grid_size - x.max() - 1) // 2
        offset_y = (self.grid_size - y.max() - 1) // 2
        offset_z = (self.grid_size - z.max() - 1) // 2
        x += offset_x
        y += offset_y
        z += offset_z

        # Ensure valid coordinates.
        mask = (x >= 0) & (x < self.grid_size) & (y >= 0) & (y < self.grid_size) & (z >= 0) & (z < self.grid_size)
        x, y, z = x[mask], y[mask], z[mask]
        pts = pts[:, mask]
        self.valid_normals = self.normals[mask]

        # Create occupancy grid.
        geom = np.zeros((5, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        geom[:, z, y, x] = pts
        self.current_object = {
            'geom': torch.FloatTensor(geom).unsqueeze(0).to(self.device),
            'coords': (x, y, z)
        }

        if self.precompute:
            print("Precomputing ContactDB heatmaps...")
            aggregated_heatmap = np.zeros(len(x))
            for _ in tqdm(range(10)):
                torch.seed()
                aggregated_heatmap += self._predict()[0]
            aggregated_heatmap /= 10.0
            self.cache = {
                "points": np.array(mesh.vertices),
                "heatmap": aggregated_heatmap,
                "normals": self.valid_normals
            }

    def _predict(self):
        with torch.no_grad():
            predictions = self.model(self.current_object['geom'])
        predictions = predictions.cpu().numpy().squeeze()
        x, y, z = self.current_object['coords']
        binary_pred = np.argmax(predictions[0], axis=0)
        heatmap = binary_pred[z, y, x].astype(float)
        heatmap = heatmap / (np.sum(heatmap) + 1e-9)
        pts = np.array(self.vertices)
        normals = self.valid_normals
        return heatmap, pts, normals

    def _get_cached(self):
        return self.cache['heatmap'], self.cache['points'], self.cache['normals']

class ContactGenHeuristic(ContactHeuristic):
    def __init__(self,
                 checkpoint_path="/home/bowenj/Projects/DexFun/third_parties/contactgen/checkpoint/checkpoint.pt",
                 cfg_path="/home/bowenj/Projects/DexFun/third_parties/contactgen/contactgen/configs/default.yaml",
                 precompute=True):
        super().__init__(precompute=precompute)
        cfg = Config(default_cfg_path=cfg_path)
        self.model = ContactGenModel(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()

    def load_object(self, mesh, n_points=8192):
        samples = trimesh.sample.sample_surface(mesh, n_points)
        verts = samples[0].astype(np.float32)
        normals = mesh.face_normals[samples[1]].astype(np.float32)
        self.current_object = {
            'verts': torch.from_numpy(verts).unsqueeze(0).to(self.device),
            'normals': torch.from_numpy(normals).unsqueeze(0).to(self.device)
        }
        if self.precompute:
            print("Precomputing ContactGen heatmaps...")
            heatmaps = []
            for _ in range(20):
                torch.seed()
                np.random.seed()
                heatmap, _, _ = self._predict()
                heatmaps.append(heatmap)
            heatmap_avg = np.mean(heatmaps, axis=0)
            self.cache = {
                'heatmap': heatmap_avg,
                'pts': self.current_object['verts'][0].cpu().numpy(),
                'normals': self.current_object['normals'][0].cpu().numpy(),
            }

    def _predict(self):
        with torch.no_grad():
            contacts, _, _ = self.model.sample(self.current_object['verts'], self.current_object['normals'])
        heatmap = contacts.squeeze().cpu().numpy()
        pts = self.current_object['verts'][0].cpu().numpy()
        nrmls = self.current_object['normals'][0].cpu().numpy()
        heatmap = heatmap / (np.sum(heatmap) + 1e-9)
        return heatmap, pts, nrmls

    def _get_cached(self):
        return self.cache['heatmap'], self.cache['pts'], self.cache['normals']


###############################################################################
# ContactMixedHeuristic class that loads precomputed heatmaps
###############################################################################

class ContactMixedHeuristic(ContactHeuristic):
    def __init__(self, heatmap_dir):
        # This class only works in precompute mode and does not load any model.
        super().__init__(precompute=True)
        self.heatmap_dir = heatmap_dir
    
    def load_object(self, mesh_name):
        """
        Loads precomputed heatmap results from a pickle file.
        """
        with open(os.path.join(self.heatmap_dir, f"{mesh_name}.pkl"), "rb") as f:
            self.cache = pickle.load(f)
        # Set current_object to a dummy value since it's required for query.
        self.current_object = mesh_name
    
    def _predict(self):
        raise NotImplementedError("ContactMixedHeuristic only supports precomputed data.")
    
    def _get_cached(self):
        return self.cache['heatmap'], self.cache['pts'], self.cache['normals']

###############################################################################
# Example usage
###############################################################################

if __name__ == "__main__":
    # For demonstration purposes, assume you have a function `load_mesh`
    # that loads a mesh from a given directory.
    # Load and preprocess test mesh
    contact_heuristic = ContactMixedHeuristic(heatmap_dir="output/mixed_heatmap")

    from dexfun.mesh.load_mesh import load_mesh, trimesh_to_o3d

    mesh_dir = "/home/bowenj/Projects/DexFun/output/meshes/mesh_raw_ahg"

    for mesh_name in os.listdir(mesh_dir):
        mesh, _ = load_mesh(mesh_dir, mesh_name, mesh_format="obj")

        print("Visualizing precomputed Contact heatmap...")
        # try:
        # contact_heuristic.load_object(mesh)
        contact_heuristic.load_object(mesh_name)
        db_heatmap, pts, normals = contact_heuristic.query(visualize=True)
        # except Exception as e:
        #     print(f"ContactDB test failed with error: {e}")

    