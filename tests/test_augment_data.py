import h5py
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# --- Configuration ---
H5_FILE_PATH = 'shelf_demos/cleaned_data.h5'
AUGMENTATION_FACTOR = 4  # Creates 4 new copies per demo
IMG_KEY = 'rgb'          # The key in your H5 file for images

# --- Augmentation Logic (Per-Episode Consistency) ---
def apply_consistent_augmentation(images_np):
    """
    Applies the SAME random augmentation to all frames in a single episode.
    This prevents the video from 'flickering' wildly, which is better for 
    temporal consistency in Transformers.
    """
    # 1. Convert to Tensor (N, C, H, W) and normalize to [0,1]
    # Assuming images_np is (N, H, W, 3)
    imgs_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    
    # 2. Sample Random Parameters ONCE per episode
    
    # A. Color Jitter Params
    brightness = random.uniform(0.6, 1.4)
    contrast   = random.uniform(0.6, 1.4)
    saturation = random.uniform(0.6, 1.4)
    hue        = random.uniform(-0.05, 0.05)
    
    # B. Shift Params (Max 5% shift to be safe)
    # We apply shift on the H and W dimensions
    _, _, h, w = imgs_tensor.shape
    max_dx = w * 0.05
    max_dy = h * 0.05
    trans_x = random.uniform(-max_dx, max_dx)
    trans_y = random.uniform(-max_dy, max_dy)

    # 3. Apply to all frames in loop (or batch)
    aug_imgs = []
    for i in range(imgs_tensor.shape[0]):
        img = imgs_tensor[i]
        
        # Apply Color
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)
        
        # Apply Shift (affine)
        # angle=0, translate=(x,y), scale=1, shear=0
        img = TF.affine(img, angle=0, translate=(trans_x, trans_y), scale=1.0, shear=0)
        
        aug_imgs.append(img)
        
    # Stack back to (N, C, H, W)
    aug_tensor = torch.stack(aug_imgs)
    
    # 4. Convert back to Numpy uint8 (0-255) for H5 storage
    # Clamp is important to avoid overflow artifacts
    aug_tensor = torch.clamp(aug_tensor * 255.0, 0, 255).byte()
    
    # Permute back to (N, H, W, 3)
    aug_np = aug_tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    return aug_np

# --- Main Processing Loop ---
def generate_augmented_h5(file_path):
    print(f"Opening {file_path}...")
    
    with h5py.File(file_path, 'a') as f:
        # Get list of original demos (exclude existing augmentations)
        demo_names = [k for k in f.keys() if k.startswith('demo_') and '_aug_' not in k]
        total = len(demo_names)
        
        print(f"Found {total} original demos. Creating {AUGMENTATION_FACTOR} augmented copies per demo.")
        
        for i, demo_name in enumerate(demo_names):
            print(f"[{i+1}/{total}] Augmenting {demo_name}...", end='\r')
            
            if IMG_KEY not in f[demo_name]:
                print(f"\nSkipping {demo_name} (no {IMG_KEY} found)")
                continue
                
            # Load original RGB
            original_rgb = f[demo_name][IMG_KEY][:]
            
            # Generate Copies
            for k in range(1, AUGMENTATION_FACTOR + 1):
                aug_name = f"{demo_name}_aug_{k}"
                
                # If it already exists, delete it to ensure we write fresh data
                if aug_name in f:
                    del f[aug_name]
                
                # Create group
                aug_grp = f.create_group(aug_name)
                
                # 1. Create Augmented RGB
                aug_rgb = apply_consistent_augmentation(original_rgb)
                
                # Save compressed to save space (RGB is heavy!)
                aug_grp.create_dataset(IMG_KEY, data=aug_rgb, compression="gzip", compression_opts=4)
                
                # 2. Copy all other non-image data (Actions, States, etc.)
                for key in f[demo_name].keys():
                    if key != IMG_KEY and key != 'depth': # Skip images, copy control data
                        # We use simple copy since these are small arrays
                        f.copy(f[demo_name][key], aug_grp, name=key)

    print(f"\n\nSuccess! Dataset expanded.")
    print(f"Original: {total} demos")
    print(f"Total Now: {total * (AUGMENTATION_FACTOR + 1)} demos")

if __name__ == "__main__":
    generate_augmented_h5(H5_FILE_PATH)