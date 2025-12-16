import h5py
import numpy as np
import torch
from transformers import AutoModel
import os
import matplotlib.pyplot as plt
import cv2

# --- Configuration ---
H5_FILE_PATH = 'table_demo.h5' # <-- CHANGE THIS to your file path
DINO_MODEL_NAME = 'facebook/dinov2-base'
CLS_FEATURE_DIM = 768 # DINOv2-base CLS token dimension

SAVE_CLS_FEATURES = True
SAVE_PATCH_FEATURES = False
# --------------------------

# --- Setup DINO Model ---
# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load and freeze DINOv2 backbone
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False
    print(f"DINOv2 model loaded successfully. CLS feature dim: {CLS_FEATURE_DIM}")
except Exception as e:
    print(f"Error loading DINO model: {e}")
    exit()

# --- MODIFIED: Pre-processing Function ---
def get_dino_features(img_batch: np.ndarray):
    """
    Handles both:
    - Depth images: (N, H, W)
    - RGB images:   (N, H, W, 3)
    """

    if img_batch.ndim == 3:
        # Depth (N, H, W)
        img_tensor = torch.from_numpy(img_batch).float()      # (N, H, W)
        img_tensor = img_tensor.unsqueeze(1)                  # (N, 1, H, W)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)            # (N, 3, H, W)

    elif img_batch.ndim == 4 and img_batch.shape[-1] == 3:
        # RGB (N, H, W, 3)
        img_tensor = torch.from_numpy(img_batch).float()      # (N, H, W, 3)
        img_tensor = img_tensor.permute(0, 3, 1, 2)           # (N, 3, H, W)

    else:
        raise ValueError(f"Unsupported image shape: {img_batch.shape}")

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = dino_model(img_tensor)
    
    
    cls_ = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    patches = outputs.last_hidden_state[:, 1:, :].cpu().numpy()

    return cls_, patches


# --- MODIFIED: Main HDF5 Processing Loop ---
def process_h5_file(file_path, img_type='depth'):
    # --- NEW: Check configuration ---
    if not SAVE_CLS_FEATURES and not SAVE_PATCH_FEATURES:
        print("Error: Both SAVE_CLS_FEATURES and SAVE_PATCH_FEATURES are False.")
        print("Nothing to save. Please enable at least one.")
        return
    # -------------------------------

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

            # Read the entire depth array into memory
            depth_data = f[demo_name][img_type][:] 
            
                
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
    process_h5_file(H5_FILE_PATH, img_type='rgb')