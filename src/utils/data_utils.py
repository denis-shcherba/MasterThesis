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
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=state_tensor.device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=state_tensor.device)
    return (state_tensor - mean) / std

def denormalize_actions(normalized_actions, stats):
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=normalized_actions.device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=normalized_actions.device)
    return normalized_actions * std + mean
