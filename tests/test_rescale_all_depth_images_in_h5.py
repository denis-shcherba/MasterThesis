import h5py
from envs.utils import rescale_img
import numpy as np

H5_FILE_PATH = 'table_demo.h5' 

def crop_all_depthimgs_in_h5(file_path: str):

    print(f"\nProcessing H5 file: {file_path}")
    
    # Open the file in append/read-write mode ('a')
    with h5py.File(file_path, 'a') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')
            
            # 1. Get the depth data
            if 'depth' not in f[demo_name]:
                print(f"  [SKIPPED] {demo_name} - 'depth' key not found.")
                continue

            depth_data = f[demo_name]['depth'][:] 
            
            original_shape = depth_data.shape

            depth_data_rescaled = np.zeros((depth_data.shape[0], 96, 96), dtype=depth_data.dtype)
            for j, depth_img in enumerate(depth_data):
                depth_img_rescaled = rescale_img(depth_img, rescale_size=96)
                depth_data_rescaled[j] = depth_img_rescaled

            # 3. Replace/Overwrite the data
            del f[demo_name]['depth']

            f[demo_name].create_dataset('depth', data=depth_data_rescaled, compression="gzip")
            # saved_keys.append(f"cls_features {new_cls_features.shape}")

            # # --- MODIFIED: Updated print statement ---
            # print(f"  ✅ {demo_name}: Replaced depth {original_shape} with {', '.join(saved_keys)}.")


if __name__ == "__main__":
    crop_all_depthimgs_in_h5(H5_FILE_PATH)