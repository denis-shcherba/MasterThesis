import numpy as np
import torch
from transformers import AutoModel
from matplotlib import cm

def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj
    
def normalize_depth(depth_tensor, stats):
    # Expects depth_tensor shape: [1, H, W]
    return (depth_tensor - stats["min"]) / stats["range"]


def normalize_state(state_tensor, stats):
    """
    Normalizes a state tensor using z-score or min-max normalization.
    Only normalizes the indices specified in stats["normalize_indices"].
    """
    device = state_tensor.device
    
    # Get indices to normalize (default to all if not specified)
    normalize_indices = stats.get("normalize_indices", list(range(len(stats.get("mean", stats.get("min", []))))))
    
    # Clone the tensor so we don't modify unnormalized indices
    normalized = state_tensor.clone()
    
    if stats["method"] == "zscore":
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        std = torch.where(std == 0, torch.tensor(1e-6, device=device), std)
        
        # Only normalize specified indices
        for idx in normalize_indices:
            normalized[..., idx] = (state_tensor[..., idx] - mean[idx]) / std[idx]
        
        return normalized
        
    elif stats["method"] == "minmax":
        if "min" not in stats or "range" not in stats:
            raise ValueError("Stats must contain 'min' and 'range' for minmax mode.")
                
        min_val = torch.tensor(stats["min"], dtype=torch.float32, device=device)
        data_range = torch.tensor(stats["range"], dtype=torch.float32, device=device)
        data_range = torch.where(data_range == 0, torch.tensor(1.0, device=device), data_range)
        
        # Only normalize specified indices
        # Formula: 2 * ( (X - min) / range ) - 1
        for idx in normalize_indices:
            normalized[..., idx] = 2 * (state_tensor[..., idx] - min_val[idx]) / data_range[idx] - 1
        
        return normalized
        
    else:
        raise ValueError(f"Unknown normalization mode: {stats['method']}.")


def denormalize_actions(normalized_actions, stats):
    """
    Denormalizes a tensor (e.g., actions) back to its original scale.
    Only denormalizes the indices specified in stats["normalize_indices"].
    """
    device = normalized_actions.device
    
    # Get indices to normalize (default to all if not specified)
    normalize_indices = stats.get("normalize_indices", list(range(len(stats.get("mean", stats.get("min", []))))))
    
    # Clone the tensor so we don't modify unnormalized indices
    denormalized = normalized_actions.clone()
    
    if stats["method"] == "zscore":
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        
        # Only denormalize specified indices
        for idx in normalize_indices:
            denormalized[..., idx] = normalized_actions[..., idx] * std[idx] + mean[idx]
        
        return denormalized
        
    elif stats["method"] == "minmax":
        if "min" not in stats or "range" not in stats:
            raise ValueError("Stats must contain 'min' and 'range' for minmax mode.")
            
        min_val = torch.tensor(stats["min"], dtype=torch.float32, device=device)
        data_range = torch.tensor(stats["range"], dtype=torch.float32, device=device)

        # Only denormalize specified indices
        # Inverse Formula: ( (X' + 1) / 2 ) * range + min
        for idx in normalize_indices:
            denormalized[..., idx] = (normalized_actions[..., idx] + 1) / 2 * data_range[idx] + min_val[idx]
        
        return denormalized
    
    else:
        raise ValueError(f"Unknown normalization mode: {stats['method']}.")
    
cmap = cm.get_cmap('jet')

def _preprocess_depth_for_dino(depth_tensor_Bx1x96x96, cmap, device):
    """
    Helper function to convert a 1-ch depth tensor to a 3-ch jet-colored tensor.
    """
    # 1. Ensure input is a CPU tensor for NumPy conversion
    #    (B, 1, 96, 96) -> (B, 96, 96)
    depth_array_Bx96x96 = depth_tensor_Bx1x96x96.squeeze(1).cpu().numpy()
    batch_size = depth_array_Bx96x96.shape[0]

    # 2. Normalize each image in the batch independently to [0, 1]
    normalized_depth = np.zeros_like(depth_array_Bx96x96, dtype=np.float32)
    for i in range(batch_size):
        img = depth_array_Bx96x96[i]
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-6: # Avoid division by zero
            normalized_depth[i] = (img - min_val) / (max_val - min_val)
        # else: leave as zeros (already is)
            
    # 3. Apply colormap: (B, 96, 96) -> (B, 96, 96, 4) [RGBA]
    colored_depth_rgba = cmap(normalized_depth)
    
    # 4. Take only RGB, discard Alpha (B, 96, 96, 3)
    colored_depth_rgb = colored_depth_rgba[..., :3]
    
    # 5. Convert back to (B, 3, 96, 96) Torch tensor for DINO
    # (B, 96, 96, 3) -> (B, 3, 96, 96)
    input_tensor_3ch = torch.from_numpy(colored_depth_rgb).permute(0, 3, 1, 2).float().to(device)
    
    return input_tensor_3ch


# --- NEW EFFICIENT FUNCTION 1 ---
def get_cls_features(depth_tensor_Bx1x96x96: torch.Tensor, 
                   dino_model: AutoModel, 
                   device: torch.device, 
                   cmap: cm.ScalarMappable) -> torch.Tensor:
    """
    Converts a batch of depth tensors to DINO CLS feature vectors
    using Jet-Coloring and a pre-loaded model.
    """
    
    # 1. Preprocess: (B, 1, 96, 96) -> (B, 3, 96, 96)
    input_tensor_3ch = _preprocess_depth_for_dino(depth_tensor_Bx1x96x96, cmap, device)

    # 2. Forward through DINO (model is already on device and in eval mode)
    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        
    # 3. Extract the CLS token
    cls_features = outputs.last_hidden_state[:, 0, :] 
    
    return cls_features


# --- NEW EFFICIENT FUNCTION 2 ---
def get_patch_features(depth_tensor_Bx1x96x96: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of depth tensors to DINO Patch feature vectors
    using Jet-Coloring and a pre-loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DINO_MODEL_NAME = 'facebook/dinov2-base' 
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)  
    # 1. Preprocess: (B, 1, 96, 96) -> (B, 3, 96, 96)
    input_tensor_3ch = _preprocess_depth_for_dino(depth_tensor_Bx1x96x96, cmap, device)

    # 2. Forward through DINO (model is already on device and in eval mode)
    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        
    # 3. Extract the Patch tokens (exclude CLS token at index 0)
    patch_tokens = outputs.last_hidden_state[:, 1:, :] 
    
    return patch_tokens