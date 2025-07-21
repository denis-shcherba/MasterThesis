import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def _create_animation(fig, ax, data, title_prefix, interval_ms, cmap=None):
    """
    Helper function to create and return a FuncAnimation object for a given set of images.

    Args:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object where the image will be plotted.
        data (np.ndarray): The image data (e.g., (N, H, W) for depth, or (N, H, W, C) for RGB).
        title_prefix (str): Prefix for the plot title (e.g., "demo_0 - Depth").
        interval_ms (int): The delay between frames in milliseconds.
        cmap (str, optional): Colormap for the image (e.g., 'gray' for depth). Defaults to None.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object, or None if no frames.
    """
    if data.shape[0] == 0:
        print(f"Warning: No frames to animate for {title_prefix}.")
        # Close the empty figure if it was created
        plt.close(fig)
        return None

    # Initialize the image with the first frame
    im = ax.imshow(data[0], cmap=cmap)
    ax.set_title(f'{title_prefix} - Image 1')
    ax.axis('off')

    # Add colorbar only if a colormap is used (typically for depth/grayscale)
    if cmap:
        fig.colorbar(im, ax=ax, label='Value')

    def update(frame):
        """
        Update function called for each frame of the animation.
        """
        im.set_array(data[frame])
        ax.set_title(f'{title_prefix} - Image {frame + 1}')
        return [im] # Return the list of artists that were modified for blitting

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(data), interval=interval_ms, blit=True, repeat=False)
    return ani

def play_h5_images_as_video_sequential(h5_file_path, interval_ms=100):
    """
    Iterates through each 'demo_X' group in an HDF5 file and plays
    'depth' and/or 'rgb' images as separate video animations,
    opening them one by one. Each video will appear after the previous one
    is closed by the user.

    Args:
        h5_file_path (str): The path to the HDF5 file.
        interval_ms (int): The delay between frames in milliseconds.
                           Default is 100ms (10 frames per second).
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            found_any_video = False
            for key in f.keys():
                if key.startswith('demo_') and isinstance(f[key], h5py.Group):
                    demo_group = f[key]
                    print(f"\nProcessing group: {key}")

                    # Play depth images as video
                    if 'depth' in demo_group:
                        depth_data = demo_group['depth'][:]
                        print(f"  Found depth data with shape: {depth_data.shape}")
                        
                        fig_depth, ax_depth = plt.subplots(figsize=(8, 6))
                        ani_depth = _create_animation(fig_depth, ax_depth, depth_data,
                                                      f'{key} - Depth', interval_ms, cmap='gray')
                        if ani_depth: # Only show if animation was successfully created
                            print(f"  Playing {key} - Depth video. Close window to proceed.")
                            found_any_video = True
                            plt.show() # This call blocks until the figure is closed
                        else:
                            print(f"  Skipping {key} - Depth (no frames).")
                    else:
                        print(f"  No 'depth' dataset found in '{key}'.")

                    # Play RGB images as video
                    if 'rgb' in demo_group:
                        rgb_data = demo_group['rgb'][:]
                        print(f"  Found RGB data with shape: {rgb_data.shape}")
                        # Ensure RGB data is in a plotable format (Height, Width, Channels)
                        # If your RGB data is (N, C, H, W), you might need to transpose:
                        # rgb_data = np.transpose(rgb_data, (0, 2, 3, 1))
                        # Matplotlib expects float values between 0.0 and 1.0 or int values between 0-255.

                        fig_rgb, ax_rgb = plt.subplots(figsize=(8, 6))
                        ani_rgb = _create_animation(fig_rgb, ax_rgb, rgb_data,
                                                    f'{key} - RGB', interval_ms) # No cmap for RGB
                        if ani_rgb: # Only show if animation was successfully created
                            print(f"  Playing {key} - RGB video. Close window to proceed.")
                            found_any_video = True
                            plt.show() # This call blocks until the figure is closed
                        else:
                            print(f"  Skipping {key} - RGB (no frames).")
                    else:
                        print(f"  No 'rgb' dataset found in '{key}'.")

                else:
                    print(f"Skipping non-demo group/dataset: {key}")
            
            if not found_any_video:
                print("\nNo suitable image data found to create any animations.")

    except FileNotFoundError:
        print(f"Error: H5 file not found at '{h5_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (dummy file creation):
if __name__ == "__main__":
    # Create a dummy H5 file for demonstration purposes
    dummy_h5_file_name = 'variable_demo.h5'

    # 200ms = 5 frames/sec
    # 50ms = 20 frames/sec
    play_h5_images_as_video_sequential(dummy_h5_file_name, interval_ms=100)