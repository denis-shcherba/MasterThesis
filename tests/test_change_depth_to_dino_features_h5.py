import h5py
import numpy as np
import torch
from transformers import AutoModel
import os

# --- Configuration ---
H5_FILE_PATH = 'table_demo_3posrel.h5' # <-- CHANGE THIS to your file path
DINO_MODEL_NAME = 'facebook/dinov2-base'
CLS_FEATURE_DIM = 768 # DINOv2-base CLS token dimension

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

# --- Pre-processing Function ---
def get_cls_features(depth_array_64x96x96: np.ndarray) -> np.ndarray:
    """
    Converts a batch of depth numpy arrays to DINO CLS feature vectors.
    
    Args:
        depth_array_64x96x96: (64, 96, 96) numpy array of depth images (already scaled/normalized).
        
    Returns:
        (64, 768) numpy array of DINO CLS features.
    """
    # 1. Convert to Torch Tensor and add Channel/Batch dims (64, 96, 96) -> (64, 1, 96, 96)
    # Convert to float and normalize if necessary (assuming your original data is [0, 1] normalized)
    depth_tensor_1ch = torch.from_numpy(depth_array_64x96x96).float().unsqueeze(1)
    
    # 2. Replicate to 3 channels (64, 1, 96, 96) -> (64, 3, 96, 96) and move to device
    input_tensor_3ch = depth_tensor_1ch.repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        
    # 3. Extract the CLS token (index 0 of the sequence dimension)
    # Shape: (Batch_size, Num_Tokens + 1, Hidden_Size) -> (Batch_size, Hidden_Size)
    cls_features = outputs.last_hidden_state[:, 0, :] 
    
    # 4. Convert back to CPU NumPy array
    return cls_features.cpu().numpy()

# --- Main HDF5 Processing Loop ---
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
            
            # Ensure depth array has the expected shape (64, 96, 96)
            if depth_data.ndim != 3 or depth_data.shape[1] != 96 or depth_data.shape[2] != 96:
                print(f"  [ERROR] {demo_name} - Unexpected depth shape: {depth_data.shape}")
                continue

            # 2. Generate DINO CLS Features
            new_cls_features = get_cls_features(depth_data)

            # 3. Replace/Overwrite the data
            
            # a) Delete the old 'depth' dataset
            del f[demo_name]['depth']
            
            # b) Create a new 'cls_features' dataset
            # New shape will be (64, 768)
            f[demo_name].create_dataset('cls_features', data=new_cls_features, compression="gzip")
            
            # The 'path' array (64, 3) remains untouched.
            
            print(f"  ✅ {demo_name}: Replaced depth (64x96x96) with cls_features (64x{CLS_FEATURE_DIM}).")

    print("\n\nProcessing complete! New feature key is 'cls_features'.")

if __name__ == "__main__":
    process_h5_file(H5_FILE_PATH)