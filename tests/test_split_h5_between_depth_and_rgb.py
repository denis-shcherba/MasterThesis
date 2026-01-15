import h5py
from tqdm import tqdm

def split_modalities(input_path, rgb_out_path, depth_out_path):
    with h5py.File(input_path, 'r') as src, \
         h5py.File(rgb_out_path, 'w') as f_rgb, \
         h5py.File(depth_out_path, 'w') as f_depth:

        # Loop through each demo (demo_0, demo_1, etc.)
        for demo_name in tqdm(src.keys(), desc="Splitting H5"):
            demo_grp = src[demo_name]
            
            # Create the demo folder in both output files
            rgb_grp = f_rgb.create_group(demo_name)
            depth_grp = f_depth.create_group(demo_name)

            # Iterate through all arrays inside the demo
            for key in demo_grp.keys():
                # Logic for File A: Everything EXCEPT depth
                if key != 'depth':
                    src.copy(f"{demo_name}/{key}", rgb_grp, name=key)
                
                # Logic for File B: Everything EXCEPT rgb
                if key != 'rgb':
                    src.copy(f"{demo_name}/{key}", depth_grp, name=key)

    print(f"\nSuccess!")
    print(f"File A (RGB + States): {rgb_out_path}")
    print(f"File B (Depth + States): {depth_out_path}")

# Usage
split_modalities('shelf_demos/shelf_demo.h5', 'shelf_demos/shelf_demo_rgb.h5', 'shelf_demos/shelf_demo_depth.h5')