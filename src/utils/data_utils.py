import numpy as np
import torch
from transformers import AutoModel
import robotic as ry
from envs.utils import sample_points, grounded_segmentation

def get_sam_pointcloud(C: ry.Config, camera: str, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    labels = ["red cuboid"]
    threshold = 0.3

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    _, detections = grounded_segmentation(
        image=rgb,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    print("Detections:", len(detections))

    mask = detections[0].mask  
    mask = mask.astype(bool)
    masked_depth = depth.copy()
    masked_depth[~mask] = 0

    CameraView = ry.CameraView(C)
    CameraView = ry.CameraView(C)
    CameraView.setCamera(C.getFrame(camera))
    fx, fy, cx, cy = CameraView.getFxycxy()
    point_cloud = ry.depthImage2PointCloud(masked_depth, [fx, fy, cx, cy])

    points = point_cloud.reshape(-1, 3) 
    points = points[~np.all(points == 0, axis=1)]
    points = sample_points(points, n_samples=4096) 

    return points


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


def get_patch_features(depth_array_1x96x96: np.ndarray) -> np.ndarray:
    """
    Converts a batch of depth images (1×96×96) to DINO patch feature grids.
    
    Args:
        depth_array_1x96x96: numpy array of shape (B, 1, 96, 96)
            Depth images already scaled/normalized to [0, 1].
    
    Returns:
        patch_features: numpy array of shape (B, Num_Patches, 768)
            DINO patch features (excluding CLS token).
            For DINOv2-base, 96x96 input yields 36 patches (6x6 grid).
    """
    DINO_MODEL_NAME = "facebook/dinov2-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained DINO model
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    # Prepare input tensor: (B, 1, 96, 96) → (B, 3, 96, 96)
    if isinstance(depth_array_1x96x96, np.ndarray):
        depth_tensor = torch.from_numpy(depth_array_1x96x96).float()
    else:
        depth_tensor = depth_array_1x96x96.float()

    input_tensor_3ch = depth_tensor.repeat(1, 3, 1, 1).to(device)

    # Forward through DINO
    with torch.no_grad():
        outputs = dino_model(input_tensor_3ch)
        # Exclude CLS token → keep only patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, Num_Patches, 768)

    return patch_tokens