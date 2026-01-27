import h5py
import os

def shorten_h5(input_file, output_file, num_demos_to_keep, compression='gzip'):
    """
    Copies the first N demos from input_file to output_file with compression.
    """
    with h5py.File(input_file, 'r') as f_in:
        # Get all demo keys and sort them to ensure we take demo_0, demo_1, etc.
        # This assumes your naming convention allows for simple alphabetical sorting.
        all_demos = sorted(list(f_in.keys()), key=lambda x: int(x.split('_')[1]) if '_' in x else x)
        
        demos_to_copy = all_demos[:num_demos_to_keep]
        
        print(f"Found {len(all_demos)} total demos.")
        print(f"Copying the first {num_demos_to_keep} demos to {output_file}...")

        with h5py.File(output_file, 'w') as f_out:
            for demo_name in demos_to_copy:
                # Use copy() to move the entire group and its children (observations/paths)
                f_in.copy(demo_name, f_out)
                
        print("Success! Repacking complete.")

# --- Settings ---
source_h5 = 'table_demo_r.h5'
target_h5 = 'demos/push_cylinder_demo_5000_rgb.h5'
limit = 5000 # Change this to 250 or whatever you need

shorten_h5(source_h5, target_h5, limit)

# Compare sizes
old_size = os.path.getsize(source_h5) / (1024**2)
new_size = os.path.getsize(target_h5) / (1024**2)
print(f"Original size: {old_size:.2f} MB")
print(f"New size: {new_size:.2f} MB")