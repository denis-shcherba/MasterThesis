import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_planes(ax, plane_equations):
    for a, b, c, d in plane_equations:
        # Create grid range
        grid_range = np.linspace(-5, 5, 20)
        
        # Handle different plane orientations
        if abs(c) > 1e-10:  # Normal case: z can be calculated
            xx, yy = np.meshgrid(grid_range, grid_range)
            zz = (-a * xx - b * yy - d) / c
            
        elif abs(a) > 1e-10:  # Vertical YZ-plane (x is constant)
            yy, zz = np.meshgrid(grid_range, grid_range)
            xx = np.full_like(yy, -d / a)
            
        elif abs(b) > 1e-10:  # Vertical XZ-plane (y is constant)
            xx, zz = np.meshgrid(grid_range, grid_range)
            yy = np.full_like(xx, -d / b)
            
        else:  # Horizontal XY-plane (z is constant)
            xx, yy = np.meshgrid(grid_range, grid_range)
            zz = np.full_like(xx, -d / c) if c != 0 else 0
        
        # Random color
        color = np.random.rand(3,)
        
        # Plot the surface
        ax.plot_surface(xx, yy, zz, alpha=0.5, color=color)

def plot_point_cloud(ax, point_cloud, max_points=10000, sample_method='random'):
    """
    Plots a subsampled point cloud.

    :param ax: Matplotlib 3D axis object.
    :param point_cloud: 3xN NumPy array containing X, Y, and Z coordinates.
    :param max_points: Maximum number of points to plot.
    :param sample_method: Sampling method ('random', 'uniform', 'first')
    """
    if point_cloud.shape[0] != 3:
        point_cloud = point_cloud.T  # Transpose if needed
    
    # Subsample the point cloud
    if sample_method == 'random':
        # Random sampling
        if point_cloud.shape[1] > max_points:
            indices = np.random.choice(point_cloud.shape[1], max_points, replace=False)
            subsampled_cloud = point_cloud[:, indices]
        else:
            subsampled_cloud = point_cloud
    
    elif sample_method == 'uniform':
        # Uniform subsampling
        step = max(1, point_cloud.shape[1] // max_points)
        subsampled_cloud = point_cloud[:, ::step]
    
    elif sample_method == 'first':
        # Take first max_points
        subsampled_cloud = point_cloud[:, :max_points]
    
    else:
        raise ValueError("Invalid sampling method. Choose 'random', 'uniform', or 'first'.")

    # Scatter plot with reduced point size and alpha for performance
    ax.scatter(subsampled_cloud[0], subsampled_cloud[1], subsampled_cloud[2], 
               c='red', marker='o', label="Point Cloud", alpha=0.5, s=5)

def main():
    # Load point cloud
    pcl = np.load("point_cloud.npy")
    print(f"Original point cloud shape: {pcl.shape}")

    # Create figure with increased DPI for better performance
    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Example planes
    plane1 = [1, 0, 0, -2]  # x = 2 (vertical YZ-plane)
    plane2 = [0, 1, 0, -3]  # y = 3 (vertical XZ-plane)
    plane3 = [0, 0, 1, -1]  # z = 1 (horizontal XY-plane)
    plane_equations = np.array([plane1, plane2, plane3])

    # Visualize planes
    #visualize_planes(ax, plane_equations)

    # Plot point cloud with different subsampling methods
    plot_point_cloud(ax, pcl, max_points=5000, sample_method='uniform')

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Planes with Subsampled Point Cloud")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
