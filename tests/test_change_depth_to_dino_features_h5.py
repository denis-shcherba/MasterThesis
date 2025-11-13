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

# --- NEW: Feature Selection ---
# Set these to True or False based on what you want to save.
# Example 1 (CLS only):     SAVE_CLS_FEATURES = True,  SAVE_PATCH_FEATURES = False
# Example 2 (Patch only):   SAVE_CLS_FEATURES = False, SAVE_PATCH_FEATURES = True
# Example 3 (Both):         SAVE_CLS_FEATURES = True,  SAVE_PATCH_FEATURES = True
SAVE_CLS_FEATURES = False
SAVE_PATCH_FEATURES = True
# --------------------------

cmap = cm.get_cmap('jet')
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
def get_dino_features(depth_array_64x96x96: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Converts a batch of depth numpy arrays to DINO CLS and Patch feature vectors.
    
    Args:
        depth_array_64x96x96: (Batch_size, 96, 96) numpy array of depth images.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - (Batch_size, CLS_FEATURE_DIM) numpy array of DINO CLS features.
        - (Batch_size, Num_Patches, PATCH_FEATURE_DIM) numpy array of DINO Patch features 
          (e.g., (64, 256, 768) for base model).
    """
    # 1. Convert to Torch Tensor and add Channel/Batch dims (B, 96, 96) -> (B, 1, 96, 96)
    # Convert to float and normalize if necessary (assuming your original data is [0, 1] normalized)
    depth_tensor_1ch = torch.from_numpy(depth_array_64x96x96).float().unsqueeze(1)
    
    # 2. Replicate to 3 channels (B, 1, 96, 96) -> (B, 3, 96, 96) and move to device
    input_tensor_3ch = depth_tensor_1ch.repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        
    # 3. Extract features
    # last_hidden_state shape is (Batch_size, Num_Tokens + 1, Hidden_Size)
    # e.g., (64, 257, 768) for base model (256 patches + 1 CLS)
    
    # CLS token is at index 0
    cls_features = outputs.last_hidden_state[:, 0, :] 
    
    # Patch tokens are from index 1 onwards
    patch_features = outputs.last_hidden_state[:, 1:, :]
    
    # 4. Convert back to CPU NumPy array
    return cls_features.cpu().numpy(), patch_features.cpu().numpy()

def get_dino_features_jet(depth_array_Bx96x96: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Converts a batch of depth numpy arrays to DINO CLS and Patch feature vectors
    using JET COLORMAP encoding.
    """
    
    batch_size = depth_array_Bx96x96.shape[0]
    
    # 1. Normalize each image in the batch independently to [0, 1]
    # This is CRITICAL for the colormap to work consistently
    normalized_depth = np.zeros_like(depth_array_Bx96x96, dtype=np.float32)
    for i in range(batch_size):
        img = depth_array_Bx96x96[i]
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 1e-6: # Avoid division by zero
            normalized_depth[i] = (img - min_val) / (max_val - min_val)
        # else: leave as zeros
            
    # 2. Apply colormap
    # cmap(normalized_depth) returns (B, 96, 96, 4) [RGBA]
    colored_depth_rgba = cmap(normalized_depth)
    
    # 3. Take only RGB, discard Alpha (B, 96, 96, 3)
    colored_depth_rgb = colored_depth_rgba[..., :3] # Ellipsis ... means "all other dims"
    
    # 4. Convert to (B, 3, 96, 96) Torch tensor for DINO
    # (B, 96, 96, 3) -> (B, 3, 96, 96)
    input_tensor_3ch = torch.from_numpy(colored_depth_rgb).permute(0, 3, 1, 2).float().to(device)

    # 5. Get DINO features (same as your code)
    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
    
    cls_features = outputs.last_hidden_state[:, 0, :]
    patch_features = outputs.last_hidden_state[:, 1:, :]
    
    return cls_features.cpu().numpy(), patch_features.cpu().numpy()

def get_dino_features_hha_lite(depth_array_Bx96x96: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Converts a batch of depth numpy arrays to DINO CLS and Patch feature vectors
    using HHA-Lite (Depth, Gradient-X, Gradient-Y) encoding.
    """
    
    batch_size = depth_array_Bx96x96.shape[0]
    
    # Output will be (B, 3, 96, 96)
    output_hha_lite_batch = np.zeros((batch_size, 3, 96, 96), dtype=np.float32)

    for i in range(batch_size):
        # Use the original, non-normalized depth for gradient calculation
        img = depth_array_Bx96x96[i].astype(np.float32)
        
        # --- Channel 1: Normalized Depth ---
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-6:
            norm_depth = (img - min_val) / (max_val - min_val)
        else:
            norm_depth = np.zeros((96, 96), dtype=np.float32)

        # --- Channel 2: Normalized Gradient X ---
        sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        min_val, max_val = sobel_x.min(), sobel_x.max()
        if max_val - min_val > 1e-6:
            sobel_x = (sobel_x - min_val) / (max_val - min_val)
        else:
            sobel_x = np.zeros((96, 96), dtype=np.float32)
            
        # --- Channel 3: Normalized Gradient Y ---
        sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        min_val, max_val = sobel_y.min(), sobel_y.max()
        if max_val - min_val > 1e-6:
            sobel_y = (sobel_y - min_val) / (max_val - min_val)
        else:
            sobel_y = np.zeros((96, 96), dtype=np.float32)

        # --- Stack channels ---
        output_hha_lite_batch[i, 0, :, :] = norm_depth
        output_hha_lite_batch[i, 1, :, :] = sobel_x
        output_hha_lite_batch[i, 2, :, :] = sobel_y

    # 4. Convert to Torch tensor
    input_tensor_3ch = torch.from_numpy(output_hha_lite_batch).float().to(device)

    # 5. Get DINO features (same as your code)
    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
    
    cls_features = outputs.last_hidden_state[:, 0, :]
    patch_features = outputs.last_hidden_state[:, 1:, :]
    
    return cls_features.cpu().numpy(), patch_features.cpu().numpy()

# --- MODIFIED: Main HDF5 Processing Loop ---
def process_h5_file(file_path):
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