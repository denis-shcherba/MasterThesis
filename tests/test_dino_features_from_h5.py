import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import torch

# --- Configuration ---
FILE_PATH = "table_demo.h5"  # <--- Change this to your .h5 file path
DEMO_NAME = "demo_0"                # <--- The group name (demo_x)
FRAME_IDX = 76                     # <--- The j-th frame you want to visualize
# ---------------------

def load_image_from_h5(file_path, demo_name, idx):
    try:
        with h5py.File(file_path, 'r') as f:
            if demo_name not in f:
                raise ValueError(f"Group '{demo_name}' not found in {file_path}")
            
            group = f[demo_name]
            
            if 'rgb' not in group:
                raise ValueError(f"Dataset 'rgb' not found in group '{demo_name}'")
            
            # Assuming structure is [N_frames, H, W, 3] or similar
            rgb_data = group['rgb']
            
            if idx >= len(rgb_data):
                raise IndexError(f"Frame index {idx} out of bounds for dataset length {len(rgb_data)}")
                
            image = rgb_data[idx]
            
            # Ensure image is uint8 if it isn't already
            if image.max() <= 1.0 and image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
            return image
            
    except Exception as e:
        print(f"Error loading H5: {e}")
        exit(1)

# 1. Load the specific image
print(f"Loading frame {FRAME_IDX} from {DEMO_NAME}...")
input_img = load_image_from_h5(FILE_PATH, DEMO_NAME, FRAME_IDX)

# 2. Setup DINOv2
print("Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
patch_size = model.config.patch_size

# 3. Process Image
inputs = processor(images=input_img, return_tensors="pt")
batch_size, channels, img_height, img_width = inputs.pixel_values.shape
num_patches_height = img_height // patch_size
num_patches_width = img_width // patch_size

# 4. Inference
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs[0]

# 5. Extract Features
# DINOv2 usually has a CLS token at index 0, so we take 1:
patch_features_flat = last_hidden_states[:, 1:, :].squeeze(0) # Shape: [num_patches, 768]
features_np = patch_features_flat.numpy()

# 6. PCA Compression
print("Running PCA...")
pca = PCA(n_components=3)
pca.fit(features_np)
features_pca = pca.transform(features_np)

# 7. Normalize PCA for Visualization (Min-Max scaling to 0-1)
min_vals = features_pca.min(axis=0)
max_vals = features_pca.max(axis=0)
features_display = (features_pca - min_vals) / (max_vals - min_vals)

# Reshape back to 2D spatial map
feature_map_rgb = features_display.reshape(
    (num_patches_height, num_patches_width, 3)
)

# 8. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original Image
axes[0].imshow(input_img)
axes[0].set_title(f'Original Input (Frame {FRAME_IDX})')
axes[0].axis('off')

# Feature Map
axes[1].imshow(feature_map_rgb)
axes[1].set_title(f'DINO Features (PCA to 3D RGB)\nMap Size: {num_patches_height}x{num_patches_width}')
axes[1].axis('off')

plt.tight_layout()
plt.show()