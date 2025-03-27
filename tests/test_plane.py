import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_planes(plane_equations):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
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
    
    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Plane Visualization")
    
    plt.tight_layout()
    plt.show()

# Example planes
plane1 = [1, 0, 0, -2]  # x = 2 (vertical YZ-plane)
plane2 = [0, 1, 0, -3]  # y = 3 (vertical XZ-plane)
plane3 = [0, 0, 1, -1]  # z = 1 (horizontal XY-plane)
plane_equations = np.array([plane1, plane2, plane3])

visualize_planes(plane_equations)