import h5py
from tqdm import tqdm
import numpy as np

def trim_and_remove_depth(input_path, output_path, target_frames=64):
    with h5py.File(input_path, 'r') as src, \
         h5py.File(output_path, 'w') as dst:
        
        # Iterate through all demos (demo_0, demo_1, etc.)
        for demo_name in tqdm(src.keys(), desc="Processing Demos"):
            demo_grp = src[demo_name]
            new_grp = dst.create_group(demo_name)
            
            for key in demo_grp.keys():
                # 1. COMPLETELY SKIP DEPTH
                if key == 'depth':
                    continue
                
                item = demo_grp[key]
                
                # 2. PROCESS RGB (TRIM TO 64)
                if key == 'rgb':
                    try:
                        # Attempt to read only what we need
                        data = item[:target_frames]
                        new_grp.create_dataset(
                            key, 
                            data=data, 
                            compression="gzip", 
                            chunks=True
                        )
                    except Exception as e:
                        print(f"\n[Warning] Could not read RGB for {demo_name}: {e}")
                
                # 3. COPY EVERYTHING ELSE (path, agent_pos, book, etc.)
                else:
                    try:
                        # Use copy to preserve attributes/metadata for non-visual data
                        src.copy(f"{demo_name}/{key}", new_grp, name=key)
                    except Exception as e:
                        print(f"\n[Warning] Could not copy {key} for {demo_name}: {e}")

    print(f"\nFinished! New file created without depth: {output_path}")

# Run the script
trim_and_remove_depth('shelf_demos/hook_real_augmented_570.h5', 'cleaned_rgb_only_64.h5')