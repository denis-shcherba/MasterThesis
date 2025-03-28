import numpy as np
import robotic as ry

def fit_bounding_box(points):
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

# Example usage
if __name__ == "__main__":
    # Generate a sample point cloud of a partial cuboid
    
    pcl = np.load("point_cloud.npy")
    # Axis-Aligned Bounding Box
    min_point, max_point, center, dimensions = fit_bounding_box(pcl)
    print("Axis-Aligned Bounding Box:")
    print(f"Min Point: {min_point}")
    print(f"Max Point: {max_point}")
    print(f"Center: {center}")
    print(f"Dimensions: {dimensions}")
    
    C = ry.Config()

    C.addFrame("pcl").setPointCloud(pcl)
    C.view(True)
    C.addFrame("fitted_box").setShape(ry.ST.box, size=dimensions).setPosition(center).setColor([.7,.7,.7,.5])
    C.view(True)

    # Oriented Bounding Box using PCA
    oriented_corners, axes, variances = principal_component_bounding_box(pcl)
    print("\nOriented Bounding Box:")
    print("Principal Axes:")
    print(axes)
    print("Variances:", variances)