from typing import Tuple

import threading
import numpy as np
from concurrent.futures import TimeoutError
from functools import wraps

import open3d as o3d

def timeout(seconds: float) -> callable:
    """A decorator that can be used to timeout arbitrary functions after specified time.

    Example Usage
    -------------
    @timeout(2)
    def long_running_function():
        time.sleep(5)
        return "Finished"

    try:
        result = long_running_function()
        print(result)
    except TimeoutError as e:
        print(e)

    Alternatively, you can wrap another function with it as follows.
    timed_func = timeout(2)(long_running_function)

    Parameters
    ----------
    seconds : float
        The number of seconds to wait before raising a TimeoutError.

    Returns
    -------
    callable
        A function that can be used to decorate other functions with a timeout.
    """
    # raise warning, not error, if timeout <= 0
    if seconds <= 0:
        raise Warning(
            "The specified timeout index is <= 0, so the function will immediately timeout!"
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [
                TimeoutError(
                    f"Function '{func.__name__}' timed out after {seconds} seconds"
                )
            ]
            timer = threading.Timer(
                seconds,
                lambda: result.append(
                    TimeoutError(
                        f"Function '{func.__name__}' timed out after {seconds} seconds"
                    )
                ),
            )

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            timer.start()
            thread.join(seconds)
            timer.cancel()

            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator

from pydrake.geometry import Sphere, Rgba, Cylinder
from pydrake.math import RigidTransform, RotationMatrix

def add_marker(model, pos, ort=None, color=[1,0,0,1], radius=0.005, name="marker"):
    """Add a marker to the model.
    Parameters
    ----------
    model : pymanoid.robot_model.RobotModel
        The robot model to add the marker to.
    color : tuple
        The color of the marker in RGB format.
    pos : np.ndarray
        The position of the marker.
    ort : np.ndarray, optional
        The orientation of the marker, by default None.
        3 x 3 rotation matrix
    radius : float, optional
        The radius of the marker, by default 0.01.
    """
    # Add the sphere marker
    model.meshcat.SetObject(
        path=name,
        shape=Sphere(radius),
        rgba=Rgba(*color),
    )
    model.meshcat.SetTransform(
        path=name,
        X_ParentPath=RigidTransform(pos)
    )
    
    if ort is None:
        return
    
    # Define axis properties
    axis_length = 0.02
    axis_radius = 0.002
    
    # Create cylinders for X, Y, Z axes
    axes_data = [
        (f"{name}_x_axis", [1, 0, 0, 1], 0),  # Red for X-axis
        (f"{name}_y_axis", [0, 1, 0, 1], 1),  # Green for Y-axis
        (f"{name}_z_axis", [0, 0, 1, 1], 2)   # Blue for Z-axis
    ]
    
    rotation_matrix = RotationMatrix(ort)
    
    for axis_name, axis_color, axis_idx in axes_data:
        # Create a cylinder for this axis
        model.meshcat.SetObject(
            path=axis_name,
            shape=Cylinder(radius=axis_radius, length=axis_length),
            rgba=Rgba(*axis_color),
        )
        
        # Default cylinder is along Z, so we need base rotations to align with X, Y, or Z
        if axis_idx == 0:  # X-axis
            base_rotation = RotationMatrix.MakeYRotation(np.pi/2)
        elif axis_idx == 1:  # Y-axis
            base_rotation = RotationMatrix.MakeXRotation(-np.pi/2)
        else:  # Z-axis
            base_rotation = RotationMatrix()
        
        # Apply frame orientation to base rotation
        final_rotation = rotation_matrix @ base_rotation
        
        # Position cylinder center at half the length along the appropriate axis
        offset = (axis_length / 2) * ort[:, axis_idx]
        axis_pos = pos + offset
        
        # Set the transform using rotation and position
        model.meshcat.SetTransform(
            path=axis_name,
            X_ParentPath=RigidTransform(final_rotation, axis_pos)
        )

def make_rotation_for_direction(d_vec: np.ndarray, axis: str = 'z') -> RotationMatrix:
    """Creates a rotation matrix with the specified axis aligned with d_vec and random rotation around it.
    
    Parameters
    ----------
    d_vec : np.ndarray
        The direction vector to align with
    axis : str, default='z'
        Which axis to align with d_vec. Must be 'x', 'y', or 'z'.
    
    Returns
    -------
    RotationMatrix
        A rotation matrix with the specified axis aligned with d_vec and random rotation around it.
    """
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Normalize the direction vector
    unit_vec = d_vec / np.linalg.norm(d_vec)
    
    if axis == 'x':
        # For x-axis alignment
        ref = np.array([0, 1, 0]) if abs(unit_vec[2]) > 0.9 else np.array([0, 0, 1])
        y_new = np.cross(ref, unit_vec)
        y_new /= np.linalg.norm(y_new)
        z_new = np.cross(unit_vec, y_new)
        z_new /= np.linalg.norm(z_new)
        
        # Random rotation around x
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        y_new_rot = c * y_new + s * z_new
        z_new_rot = -s * y_new + c * z_new
        
        return RotationMatrix(np.column_stack((unit_vec, y_new_rot, z_new_rot)))
        
    elif axis == 'y':
        # For y-axis alignment
        ref = np.array([1, 0, 0]) if abs(unit_vec[2]) > 0.9 else np.array([0, 0, 1])
        z_new = np.cross(unit_vec, ref)
        z_new /= np.linalg.norm(z_new)
        x_new = np.cross(z_new, unit_vec)
        x_new /= np.linalg.norm(x_new)
        
        # Random rotation around y
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        z_new_rot = c * z_new + s * x_new
        x_new_rot = -s * z_new + c * x_new
        
        return RotationMatrix(np.column_stack((x_new_rot, unit_vec, z_new_rot)))
        
    else:  # axis == 'z'
        # For z-axis alignment
        ref = np.array([1, 0, 0]) if abs(unit_vec[1]) > 0.9 else np.array([0, 1, 0])
        x_new = np.cross(ref, unit_vec)
        x_new /= np.linalg.norm(x_new)
        y_new = np.cross(unit_vec, x_new)
        y_new /= np.linalg.norm(y_new)
        
        # Random rotation around z
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        x_new_rot = c * x_new - s * y_new
        y_new_rot = s * x_new + c * y_new
        
        return RotationMatrix(np.column_stack((x_new_rot, y_new_rot, unit_vec)))
    
def find_opposite_contacts(
    contact_pos: np.ndarray,
    contact_normal: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    angle_tol: float = np.pi/6,  # 30 degrees in radians
    min_dist: float = 0.02,      # Minimum distance to consider
    max_dist: float = 0.15,      # Maximum distance to consider
    visualize: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find points that could serve as opposite contacts for a stable grasp.
    
    Parameters
    ----------
    contact_pos : np.ndarray (3,)
        Position of the palm contact point
    contact_normal : np.ndarray (3,)
        Normal at the palm contact point
    points : np.ndarray (N, 3)
        All possible contact points
    normals : np.ndarray (N, 3)
        Normals at each point
    angle_tol : float
        Tolerance angle in radians for alignment
    min_dist : float
        Minimum distance between contacts to consider
    max_dist : float
        Maximum distance between contacts to consider
    visualize : bool, default=False
        Whether to show visualization of points and results
        
    Returns
    -------
    valid_points : np.ndarray (M, 3)
        Points that satisfy the constraints
    valid_normals : np.ndarray (M, 3)
        Normals at the valid points
    distances : np.ndarray (M,)
        Distances from contact_pos to each valid point
    """
    # Calculate vectors from contact to all points
    vectors = points - contact_pos[None, :]
    
    # Calculate distances
    distances = np.linalg.norm(vectors, axis=1)
    
    # Normalize vectors
    vectors_normalized = vectors / distances[:, None]
    
    # Calculate angles with contact normal
    angles_with_contact = np.arccos(np.clip(
        np.abs(np.dot(vectors_normalized, contact_normal)),
        -1.0, 1.0
    ))
    
    # Calculate angles with point normals
    angles_with_normals = np.arccos(np.clip(
        np.abs(np.einsum('ij,ij->i', vectors_normalized, normals)),
        -1.0, 1.0
    ))
    
    # Find points that satisfy all constraints
    valid_mask = (
        (angles_with_contact <= angle_tol) &        # Aligned with contact normal
        (angles_with_normals <= angle_tol) &        # Aligned with point normal
        (distances >= min_dist) &                   # Not too close
        (distances <= max_dist)                     # Not too far
    )
    
    valid_points = points[valid_mask]
    valid_normals = normals[valid_mask]
    valid_distances = distances[valid_mask]

    if visualize:
        import open3d as o3d
        
        # Create point cloud for all points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color all points light gray
        colors = np.ones((len(points), 3)) * 0.7  # Light gray
        
        # Color valid points blue
        colors[valid_mask] = [0, 0, 1]  # Blue
        
        # Add palm contact point as a separate sphere
        contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        contact_sphere.translate(contact_pos)
        contact_sphere.paint_uniform_color([1, 0, 0])
        # contact_sphere.compute_vertex_normals()
        
        # Set colors and visualize
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd, contact_sphere])
    
    if len(valid_points) == 0:
        return None
    else:
        return valid_points, valid_normals, valid_distances