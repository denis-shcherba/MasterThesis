import numpy as np
import torch
from transformers import AutoModel

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
    
def get_cls_features(depth_array_1x96x96: np.ndarray) -> np.ndarray:
    """
    Converts a batch of depth numpy arrays to DINO CLS feature vectors.
    
    Args:
        depth_array_64x96x96: (64, 96, 96) numpy array of depth images (already scaled/normalized).
        
    Returns:
        (64, 768) numpy array of DINO CLS features.
    """
    DINO_MODEL_NAME = 'facebook/dinov2-base'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False
    # 1. Convert to Torch Tensor and add Channel/Batch dims (64, 96, 96) -> (64, 1, 96, 96)
    # Convert to float and normalize if necessary (assuming your original data is [0, 1] normalized)    
    # 2. Replicate to 3 channels (64, 1, 96, 96) -> (64, 3, 96, 96) and move to device
    input_tensor_3ch = depth_array_1x96x96.repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        
    # 3. Extract the CLS token (index 0 of the sequence dimension)
    # Shape: (Batch_size, Num_Tokens + 1, Hidden_Size) -> (Batch_size, Hidden_Size)
    cls_features = outputs.last_hidden_state[:, 0, :] 
    
    # 4. Convert back to CPU NumPy array
    return cls_features