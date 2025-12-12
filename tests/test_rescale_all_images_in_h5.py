import h5py
from envs.utils import rescale_img
import numpy as np

H5_FILE_PATH = 'table_demo.h5' 

def rescale_all_imgs_in_h5(file_path: str, img_type: str = 'depth'):

    print(f"\nProcessing H5 file: {file_path}")
    
    # Open the file in append/read-write mode ('a')
    with h5py.File(file_path, 'a') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')
            
            # 1. Get the depth data
            if img_type not in f[demo_name]:
                print(f"  [SKIPPED] {demo_name} - '{img_type}' key not found.")
                continue

            depth_data = f[demo_name][img_type][:] 
            
            original_shape = depth_data.shape

            depth_data_rescaled = np.zeros((depth_data.shape[0], 96, 96), dtype=depth_data.dtype)
            for j, depth_img in enumerate(depth_data):
                depth_img_rescaled = rescale_img(depth_img, rescale_size=96)
                depth_data_rescaled[j] = depth_img_rescaled

            # 3. Replace/Overwrite the data
            del f[demo_name][img_type]

            f[demo_name].create_dataset(img_type, data=depth_data_rescaled, compression="gzip")
            # saved_keys.append(f"cls_features {new_cls_features.shape}")

            # # --- MODIFIED: Updated print statement ---
            # print(f"  ✅ {demo_name}: Replaced depth {original_shape} with {', '.join(saved_keys)}.")

def crop_all_imgs_in_h5(file_path: str, img_type: str = 'rgb'):
    print(f"\nProcessing H5 file: {file_path}")
    
    with h5py.File(file_path, 'a') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})", end='\r')

            # Validate key
            if img_type not in f[demo_name]:
                print(f"\n  [SKIPPED] {demo_name}: key '{img_type}' not found.")
                continue

            # Load images (e.g., (N, H, W) or (N, H, W, 3))
            img_data = f[demo_name][img_type][:]
            original_shape = img_data.shape

            # Deduce shape
            if img_data.ndim == 4:
                # RGB or multi-channel: (N, H, W, C)
                N, H, W, C = img_data.shape
                cropped = img_data[:, 100:, :, :]   # top 100 rows
            elif img_data.ndim == 3:
                # Depth / grayscale: (N, H, W)
                N, H, W = img_data.shape
                cropped = img_data[:, 100:, :]      # top 100 rows
            else:
                print(f"\n  [SKIPPED] {demo_name}: unsupported image shape {img_data.shape}")
                continue

            # Replace dataset
            del f[demo_name][img_type]
            f[demo_name].create_dataset(img_type, data=cropped, compression="gzip")

        print("\nDone.")

if __name__ == "__main__":
    crop_all_imgs_in_h5(H5_FILE_PATH, img_type='rgb')