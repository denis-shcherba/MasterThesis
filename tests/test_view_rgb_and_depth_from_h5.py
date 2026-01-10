import h5py
import numpy as np
import matplotlib.pyplot as plt

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view_rgb_sequence_with_two_sliders(h5_path="push_cylinder_demo_1000_rgb.h5", rgb_keys=("rgb","color","image","images")):
    
    # 1. First, scan the file to get all demo names sorted naturally
    with h5py.File(h5_path, 'r') as f:
        demos = [k for k in f.keys() if k.startswith('demo_')]
        # Sort numerically (demo_1, demo_2, ..., demo_10) instead of alphabetically
        demos.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x)
        
    if not demos:
        print("No demo_* groups found.")
        return

    # Internal state to hold current RGB data
    # We use a mutable container (list) so the inner functions can update it
    state = {
        "rgb": None,
        "T": 0,
        "demo_idx": 0
    }

    # Helper to load a specific demo index
    def load_demo_data(index):
        demo_name = demos[index]
        with h5py.File(h5_path, 'r') as f:
            grp = f[demo_name]
            rgb_key = next((k for k in rgb_keys if k in grp), None)
            
            if rgb_key is None:
                # Return a dummy black frame if key is missing to prevent crash
                return np.zeros((1, 100, 100, 3), dtype=np.uint8), demo_name, "NONE"
            
            data = grp[rgb_key][:]
            if data.ndim == 3: data = data[None, ...] # (H,W,C) -> (1,H,W,C)
            if data.shape[-1] == 4: data = data[..., :3] # Drop alpha
            if data.dtype != np.uint8: data = np.clip(data, 0, 255).astype(np.uint8)
            
            return data, demo_name, rgb_key

    # Load initial demo (index 0)
    state["rgb"], current_demo_name, current_key = load_demo_data(0)
    state["T"], H, W, _ = state["rgb"].shape

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.3) # More space at bottom for 2 sliders

    im = ax.imshow(state["rgb"][0], origin='upper')
    title = ax.set_title(f"{current_demo_name}:{current_key} (Frame 0/{state['T']-1})")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    # Set fixed limits based on first frame (assuming all videos are same resolution)
    ax.set_xlim(0, W-1)
    ax.set_ylim(H-1, 0)

    # --- Sliders ---
    # 1. Demo Slider (Bottom)
    ax_demo = plt.axes([0.15, 0.05, 0.7, 0.04])
    slider_demo = Slider(ax_demo, 'Demo ID', 0, len(demos)-1, valinit=0, valstep=1)

    # 2. Frame Slider (Top)
    ax_frame = plt.axes([0.15, 0.12, 0.7, 0.04])
    slider_frame = Slider(ax_frame, 'Frame', 0, state["T"]-1, valinit=0, valstep=1)

    # --- Callbacks ---
    
    def update_frame(val):
        """Updates the image based on frame slider."""
        frame_idx = int(slider_frame.val)
        # Clamp index just in case
        frame_idx = min(frame_idx, state["T"] - 1)
        
        im.set_data(state["rgb"][frame_idx])
        
        # Update title with current demo and frame info
        d_idx = int(slider_demo.val)
        title.set_text(f"{demos[d_idx]} (Frame {frame_idx}/{state['T']-1})")
        fig.canvas.draw_idle()

    def update_demo(val):
        """Updates the loaded data based on demo slider."""
        demo_idx = int(slider_demo.val)
        
        # 1. Load new data
        new_rgb, d_name, d_key = load_demo_data(demo_idx)
        state["rgb"] = new_rgb
        state["T"] = new_rgb.shape[0]
        state["demo_idx"] = demo_idx

        # 2. Update Frame Slider limits (Dynamic range!)
        # We must update valmax and the axis limits
        slider_frame.valmax = state["T"] - 1
        slider_frame.ax.set_xlim(0, state["T"] - 1)
        
        # Reset frame to 0 so we don't go out of bounds on the new video
        slider_frame.set_val(0) 
        
        # (The set_val(0) triggers update_frame, so we don't need to call it manually)

    slider_frame.on_changed(update_frame)
    slider_demo.on_changed(update_demo)

    plt.show()

def view_depth_and_rgb_from_h5(show_grid=True):
    last_depth = None
    last_rgb = None

    with h5py.File("table_demo.h5", 'r') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')

            depth = f[demo_name]['depth'][:]      # (T, H, W)
            rgb = f[demo_name]['rgb'][:]          # (T, H, W, 3) or (T, H, W, 4)

            # Select a single frame (last)
            last_depth = depth[-1]                # (H, W)
            last_rgb = rgb[-1]                    # (H, W, 3/4)

    if last_depth is None or last_rgb is None:
        print("No demo_* groups found in table_demo.h5")
        return

    # Prepare depth for visualization
    depth_img = np.nan_to_num(last_depth, nan=0.0, posinf=0.0, neginf=0.0)
    finite_vals = depth_img[np.isfinite(depth_img)]
    vmin = float(np.min(finite_vals)) if finite_vals.size > 0 else None
    vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else None

    # Prepare RGB
    rgb_img = last_rgb
    if rgb_img.dtype != np.uint8:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    if rgb_img.shape[-1] == 4:
        rgb_img = rgb_img[..., :3]  # drop alpha if present

    H, W = depth_img.shape
    cx, cy = W // 2, H // 2  # center coordinates

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axd, axr = axes

    # Depth plot with axes, center crosshair, and grid
    axd.set_title("Depth")
    axd.imshow(depth_img, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    if show_grid:
        axd.grid(color='w', alpha=0.2, linestyle='-', linewidth=0.5)
    # Crosshair at center
    axd.axhline(y=cy, color='r', linestyle='--', linewidth=1)
    axd.axvline(x=cx, color='r', linestyle='--', linewidth=1)
    # Show pixel coords
    axd.set_xlabel("X (pixels)")
    axd.set_ylabel("Y (pixels)")
    axd.set_xlim(0, W-1)
    axd.set_ylim(H-1, 0)  # origin='upper' makes y inverted; keep top-left as (0,0)
    axd.tick_params(axis='both', which='both', labelsize=8)

    # RGB plot with axes, center crosshair, and grid
    axr.set_title("RGB")
    axr.imshow(rgb_img, origin='upper')
    if show_grid:
        axr.grid(color='w', alpha=0.2, linestyle='-', linewidth=0.5)
    axr.axhline(y=cy, color='r', linestyle='--', linewidth=1)
    axr.axvline(x=cx, color='r', linestyle='--', linewidth=1)
    axr.set_xlabel("X (pixels)")
    axr.set_ylabel("Y (pixels)")
    axr.set_xlim(0, W-1)
    axr.set_ylim(H-1, 0)
    axr.tick_params(axis='both', which='both', labelsize=8)

    plt.tight_layout()
    plt.show()

            
if __name__ == "__main__":
    view_rgb_sequence_with_two_sliders()