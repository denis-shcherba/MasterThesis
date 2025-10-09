import numpy as np
import torch

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
    """
    device = state_tensor.device
    
    if stats["method"] == "zscore":
        # ... (your z-score logic is correct)
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        std = torch.where(std == 0, torch.tensor(1e-6, device=device), std)
        return (state_tensor - mean) / std
        
    elif stats["method"] == "minmax":
        # 🐛 FIX: This now normalizes to [-1, 1] instead of [0, 1]
        if "min" not in stats or "range" not in stats:
            raise ValueError("Stats must contain 'min' and 'range' for minmax mode.")
                
        min_val = torch.tensor(stats["min"], dtype=torch.float32, device=device)
        data_range = torch.tensor(stats["range"], dtype=torch.float32, device=device)
        data_range = torch.where(data_range == 0, torch.tensor(1.0, device=device), data_range)
        
        # Formula: 2 * ( (X - min) / range ) - 1
        return 2 * (state_tensor - min_val) / data_range - 1
        
    else:
        raise ValueError(f"Unknown normalization mode: {stats['method']}.")


def denormalize_actions(normalized_actions, stats):
    """
    Denormalizes a tensor (e.g., actions) back to its original scale.
    """
    device = normalized_actions.device
    
    if stats["method"] == "zscore":
        # ... (your z-score logic is correct)
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        return normalized_actions * std + mean
        
    elif stats["method"] == "minmax":
        # 🐛 FIX: This now denormalizes from [-1, 1]
        if "min" not in stats or "range" not in stats:
            raise ValueError("Stats must contain 'min' and 'range' for minmax mode.")
            
        min_val = torch.tensor(stats["min"], dtype=torch.float32, device=device)
        data_range = torch.tensor(stats["range"], dtype=torch.float32, device=device)

        # Inverse Formula: ( (X' + 1) / 2 ) * range + min
        return (normalized_actions + 1) / 2 * data_range + min_val