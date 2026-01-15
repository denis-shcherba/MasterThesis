import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

# --- Configuration ---
H5_FILE_PATH = 'shelf_demos/shelf_demo_depth.h5'
AUGMENTATION_FACTOR = 4
IMG_KEY = 'depth'  # Switch to 'rgb' if running on RGB files
IS_DEPTH = True    # Set to True to apply depth-specific noise

def apply_depth_augmentation(depth_np):
    """
    Applies noise and spatial shifts to depth data (N, H, W) or (N, H, W, 1).
    Assumes depth is in float32 (meters).
    """
    # 1. Convert to Tensor (N, 1, H, W)
    imgs_tensor = torch.from_numpy(depth_np).float()
    if imgs_tensor.ndimension() == 3:
        imgs_tensor = imgs_tensor.unsqueeze(1)
    elif imgs_tensor.shape[-1] == 1: # Handle (N, H, W, 1)
        imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)

    # 2. Sample Random Parameters ONCE per episode (Consistency)
    # A. Spatial Shift (5% max)
    _, _, h, w = imgs_tensor.shape
    max_dx, max_dy = w * 0.05, h * 0.05
    trans_x = random.uniform(-max_dx, max_dx)
    trans_y = random.uniform(-max_dy, max_dy)

    # B. Depth Noise Params
    bias = random.uniform(-0.02, 0.02)      # +/- 2cm offset
    scale = random.uniform(0.98, 1.02)     # 2% scaling error
    gauss_std = 0.003                      # 3mm precision jitter
    
    # 3. Apply Transformations
    aug_imgs = []
    for i in range(imgs_tensor.shape[0]):
        img = imgs_tensor[i]
        
        # Physical Augmentation
        img = (img * scale) + bias
        img += torch.randn_like(img) * gauss_std
        
        # Hole Injection (Simulating sensor 'blackouts')
        if random.random() > 0.5:
            for _ in range(random.randint(1, 3)):
                y, x = random.randint(0, h-20), random.randint(0, w-20)
                img[:, y:y+15, x:x+15] = 0

        # Spatial Shift (Must match RGB logic if using both)
        img = TF.affine(img, angle=0, translate=(trans_x, trans_y), scale=1.0, shear=0)
        
        aug_imgs.append(img)
        
    aug_tensor = torch.stack(aug_imgs)
    
    # Return as (N, H, W) or (N, H, W, 1) to match original format
    return aug_tensor.squeeze(1).cpu().numpy()

def generate_augmented_h5(file_path):
    print(f"Opening {file_path}...")
    
    with h5py.File(file_path, 'a') as f:
        demo_names = [k for k in f.keys() if k.startswith('demo_') and '_aug_' not in k]
        
        for i, demo_name in enumerate(demo_names):
            print(f"[{i+1}/{len(demo_names)}] Augmenting {demo_name}...", end='\r')
            
            if IMG_KEY not in f[demo_name]: continue
            original_data = f[demo_name][IMG_KEY][:]
            
            for k in range(1, AUGMENTATION_FACTOR + 1):
                aug_name = f"{demo_name}_aug_{k}"
                if aug_name in f: del f[aug_name]
                
                aug_grp = f.create_group(aug_name)
                
                # Apply appropriate logic
                if IS_DEPTH:
                    aug_data = apply_depth_augmentation(original_data)
                else:
                    # Use your existing apply_consistent_augmentation(original_data) here
                    pass 
                
                aug_grp.create_dataset(IMG_KEY, data=aug_data, compression="gzip", compression_opts=4)
                
                # Copy Actions/States (Stay the same)
                for key in f[demo_name].keys():
                    if key not in ['rgb', 'depth']:
                        f.copy(f[demo_name][key], aug_grp, name=key)

    print(f"\nSuccess! Dataset expanded.")

if __name__ == "__main__":
    generate_augmented_h5(H5_FILE_PATH)