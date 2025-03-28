import numpy as np
import open3d as o3d

def create_plane_mesh(plane_equation, size=1):
    """
    Create a small double-sided mesh representation of a plane given its equation ax + by + cz + d = 0
    """
    a, b, c, d = plane_equation
    
    # Compute plane normal and a point on the plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Create a point on the plane
    if abs(c) > 1e-10:
        point = np.array([0, 0, -d/c])
    elif abs(a) > 1e-10:
        point = np.array([-d/a, 0, 0])
    elif abs(b) > 1e-10:
        point = np.array([0, -d/b, 0])
    else:
        point = np.array([0, 0, 0])
    
    # Create two orthogonal vectors in the plane
    if np.abs(normal[0]) > np.abs(normal[1]):
        u = np.array([normal[1], -normal[0], 0])
    else:
        u = np.array([0, normal[2], -normal[1]])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Create vertices
    vertices = [
        point + size * (u + v),
        point + size * (u - v),
        point - size * (u + v),
        point - size * (u - v)
    ]
    
    # Create triangles (both front and back)
    triangles = [
        [0, 1, 2], [0, 2, 3],  # Front face
        [2, 1, 0], [3, 2, 0]   # Back face (reversed order)
    ]
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Colorize the mesh with a random color
    color = np.random.rand(3)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    
    return mesh


def create_random_point_cloud(num_points=500, range_min=-1, range_max=1):
    """
    Create a small random point cloud within a specified range.
    """
    points = np.random.uniform(range_min, range_max, size=(num_points, 3))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Color the points (optional)
    point_cloud.colors = o3d.utility.Vector3dVector(np.random.rand(num_points, 3))
    
    return point_cloud


def visualize_planes_and_point_cloud(plane_equations, point_cloud, window_name="Visualization"):
    """
    Visualize multiple small planes and a point cloud using Open3D.
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # Add plane meshes
    for plane_eq in plane_equations:
        plane_mesh = create_plane_mesh(plane_eq)
        vis.add_geometry(plane_mesh)
    
    # Add the point cloud
    vis.add_geometry(point_cloud)
    
    # Set up camera view
    view_control = vis.get_view_control()
    
    # Optional: Configure initial view
    view_control.set_zoom(1.2)
    view_control.set_front([0.4, -0.4, -0.8])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 1, 0])
    
    # Add small coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(coordinate_frame)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Run the visualization
    vis.run()
    vis.destroy_window()


# Example usage
if __name__ == "__main__":
    # Example plane equations: ax + by + cz + d = 0
    plane_equations = [
        (1, 0, 0, -0.2),   # YZ plane at x = -0.2
        (0, 1, 0, 0.3),    # XZ plane at y = 0.3
        (0, 0, 1, -0.1),   # XY plane at z = -0.1
        (1, 1, 1, 0)       # Diagonal plane passing through origin
    ]
    
    # Create a small random point cloud
    point_cloud = create_random_point_cloud(num_points=0)
    
    # Visualize planes and point cloud
    visualize_planes_and_point_cloud(plane_equations, point_cloud)
