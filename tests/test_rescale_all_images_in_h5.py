import h5py
import numpy as np
from envs.utils import rescale_img_with_padding

H5_FILE_PATH = 'table_demo.h5' 

def rescale_all_imgs_in_h5(file_path: str, img_type: str = 'depth', target_size: int = 224):
    print(f"\nProcessing H5 file: {file_path}")
    print(f"Target Mode: {img_type} | Target Size: {target_size}x{target_size} (Padded)")
    
    with h5py.File(file_path, 'a') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')
            
            if img_type not in f[demo_name]:
                print(f"  [SKIPPED] {demo_name} - '{img_type}' key not found.")
                continue

            # Load data
            original_data = f[demo_name][img_type][:] 
            N = original_data.shape[0]

            # Determine new shape (Handle RGB vs Depth automatically)
            if original_data.ndim == 4: # RGB (N, H, W, C)
                C = original_data.shape[3]
                new_shape = (N, target_size, target_size, C)
            else: # Depth (N, H, W)
                new_shape = (N, target_size, target_size)

            # Allocate new array
            data_rescaled = np.zeros(new_shape, dtype=original_data.dtype)

            # Loop and Rescale
            for j in range(N):
                img = original_data[j]
                # Apply the padded rescale
                data_rescaled[j] = rescale_img_with_padding(img, rescale_size=target_size)

            # Overwrite the dataset in the H5 file
            del f[demo_name][img_type]
            f[demo_name].create_dataset(img_type, data=data_rescaled, compression="gzip")
            
    print(f"\n✅ Done. All '{img_type}' images resized to {target_size}x{target_size} with padding.")

if __name__ == "__main__":
    # WARNING: I changed target_size to 224 because 96 is too small for DINO/ResNet!
    # If you strictly need 96, change it back, but 224 is recommended.
    rescale_all_imgs_in_h5(H5_FILE_PATH, img_type='depth', target_size=224) 
    # rescale_all_imgs_in_h5(H5_FILE_PATH, img_type='rgb', target_size=224)