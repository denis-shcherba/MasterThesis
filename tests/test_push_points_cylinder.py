# TODO

import h5py
import numpy as np
import torch
from transformers import AutoModel
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

# --- Configuration ---
H5_FILE_PATH = 'table_demo.h5' # <-- CHANGE THIS to your file path
DINO_MODEL_NAME = 'facebook/dinov2-base'
CLS_FEATURE_DIM = 768 # DINOv2-base CLS token dimension


def process_h5_file(file_path):

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

            # Read the entire depth array into memory
            depth_data = f[demo_name]['depth'][:] 
            
            # Ensure depth array has the expected shape (B, 96, 96)
            if depth_data.ndim != 3 or depth_data.shape[1] != 96 or depth_data.shape[2] != 96:
                print(f"  [ERROR] {demo_name} - Unexpected depth shape: {depth_data.shape}")
                continue
                
            original_shape = depth_data.shape

            # 2. Generate DINO Features
            # We still get both, but will only save what's requested
            new_cls_features, new_patch_features = get_dino_features(depth_data)
            # 3. Replace/Overwrite the data
            
            # a) Delete the old 'depth' dataset
            del f[demo_name]['depth']
            
            # --- NEW: Conditional Saving ---
            saved_keys = []
            
            # b) Create a new 'cls_features' dataset if requested
            if SAVE_CLS_FEATURES:
                f[demo_name].create_dataset('cls_features', data=new_cls_features, compression="gzip")
                saved_keys.append(f"cls_features {new_cls_features.shape}")
            
            # c) Create a new 'patch_features' dataset if requested
            if SAVE_PATCH_FEATURES:
                f[demo_name].create_dataset('patch_features', data=new_patch_features, compression="gzip")
                saved_keys.append(f"patch_features {new_patch_features.shape}")
            # ---------------------------------
            
            # The 'path' array remains untouched.
            
            # --- MODIFIED: Updated print statement ---
            print(f"  ✅ {demo_name}: Replaced depth {original_shape} with {', '.join(saved_keys)}.")

    # --- MODIFIED: Updated final message ---
    print(f"\n\nProcessing complete! New feature keys are: {', '.join(k.split(' ')[0] for k in saved_keys)}")

if __name__ == "__main__":
    process_h5_file(H5_FILE_PATH)