import numpy as np
#import open3d as o3d 
from scipy.spatial import ConvexHull
import math
import robotic as ry 
from utils import cuboid_corners_to_size_com


def fit_aabb(points):
    """
    Fit a minimal axis-aligned bounding box around a point cloud.
    
    Parameters:
    points (numpy.ndarray): Array of shape (N, 3) representing point cloud coordinates
    
    Returns:
    tuple: (min_point, max_point, center, dimensions)
        - min_point: Minimum coordinate of the bounding box
        - max_point: Maximum coordinate of the bounding box
        - center: Center point of the bounding box
        - dimensions: Dimensions (width, height, depth) of the bounding box
    """
    # Compute minimum and maximum points along each axis
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    
    # Calculate center and dimensions
    center = (min_point + max_point) / 2
    dimensions = max_point - min_point
    
    return min_point, max_point, center, dimensions

def principal_component_bounding_box(points):
    """
    Fit a bounding box using Principal Component Analysis (PCA)
    to handle potentially rotated or skewed point clouds.
    
    Parameters:
    points (numpy.ndarray): Array of shape (N, 3) representing point cloud coordinates
    
    Returns:
    tuple: (oriented_bbox_points, principal_axes, variances)
        - oriented_bbox_points: 8 corner points of the oriented bounding box
        - principal_axes: Rotation matrix of the principal components
        - variances: Variance along each principal component
    """
    # Center the points
    centered_points = points - np.mean(points, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_points.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sort_indices]
    variances = eigenvalues[sort_indices]
    
    # Project points onto principal axes
    projected_points = np.dot(centered_points, principal_axes)
    
    # Compute min and max for each principal component
    min_proj = np.min(projected_points, axis=0)
    max_proj = np.max(projected_points, axis=0)
    
    # Generate oriented bounding box corners
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner = (
                    min_proj[0] * principal_axes[:, 0] * (1 if i else -1) +
                    min_proj[1] * principal_axes[:, 1] * (1 if j else -1) +
                    min_proj[2] * principal_axes[:, 2] * (1 if k else -1)
                )
                corners.append(corner + np.mean(points, axis=0))
    
    return np.array(corners), principal_axes, variances

# Doesnt really work TODO?
# def min_volume_bounding_box(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     obb = pcd.get_oriented_bounding_box()
#     return np.asarray(obb.get_box_points()), obb.R, obb.extent

def minimum_bounding_box_from_convex_hull(points):
    """
    Find the minimum area oriented bounding box of a point cloud using the convex hull.
    
    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) representing 3D point cloud coordinates
    
    Returns:
        tuple: (oriented_bbox_points, rotation_matrix)
            - oriented_bbox_points: 8 corner points of the oriented bounding box
            - rotation_matrix: 3x3 rotation matrix representing the orientation
    """
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # If we have fewer than 4 points, we can't compute a proper 3D oriented box

    
    # Try each face normal of the convex hull as a potential axis
    min_volume = float('inf')
    best_box = None
    best_rotation = None
    
    for simplex in hull.simplices:
        # Get face vertices
        face_vertices = points[simplex]
        
        # Compute face normal
        v1 = face_vertices[1] - face_vertices[0]
        v2 = face_vertices[2] - face_vertices[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Find two orthogonal vectors in the plane of the face
        if abs(normal[0]) > abs(normal[1]):
            ortho1 = np.array([-normal[2], 0, normal[0]]) / math.sqrt(normal[0]**2 + normal[2]**2)
        else:
            ortho1 = np.array([0, -normal[2], normal[1]]) / math.sqrt(normal[1]**2 + normal[2]**2)
        
        ortho2 = np.cross(normal, ortho1)
        
        # Normalize vectors
        ortho1 = ortho1 / np.linalg.norm(ortho1)
        ortho2 = ortho2 / np.linalg.norm(ortho2)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((ortho1, ortho2, normal))
        
        # Project points to the new coordinate system
        rotated_points = np.dot(points, rotation_matrix)
        
        # Find min and max in each dimension
        min_coords = np.min(rotated_points, axis=0)
        max_coords = np.max(rotated_points, axis=0)
        
        # Calculate volume
        dimensions = max_coords - min_coords
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        
        if volume < min_volume:
            min_volume = volume
            
            # Create the box corners
            corners = []
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        corner = np.array([
                            min_coords[0] if i == 0 else max_coords[0],
                            min_coords[1] if j == 0 else max_coords[1],
                            min_coords[2] if k == 0 else max_coords[2]
                        ])
                        # Transform back to original coordinate system
                        corner_orig = np.dot(corner, rotation_matrix.T)
                        corners.append(corner_orig)
            
            best_box = np.array(corners)
            best_rotation = rotation_matrix

    return best_box, best_rotation