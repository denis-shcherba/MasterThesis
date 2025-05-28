import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
import logging

class DemonstrationDataset(Dataset):
    """
    Dataset class for loading demonstration data from H5 files.
    
    Each demonstration contains:
    - path: (64, 9) array representing 64 path steps with 9D pose representation
    - point_cloud: (64, 1000, 3) array representing point clouds for each path step
    """
    
    def __init__(
        self, 
        h5_file_path: str,
        transform: Optional[callable] = None,
        normalize_point_clouds: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the dataset.
        
        Args:
            h5_file_path: Path to the H5 file containing demonstrations
            transform: Optional transform to apply to point clouds
            normalize_point_clouds: Whether to normalize point clouds to unit sphere
            device: Device to load tensors on
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.normalize_point_clouds = normalize_point_clouds
        self.device = device
        
        # Load and cache demo keys
        with h5py.File(h5_file_path, 'r') as f:
            self.demo_keys = [key for key in f.keys() if key.startswith('demo_')]
            self.demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
        
        self.num_demos = len(self.demo_keys)
        self.steps_per_demo = 64  # Fixed based on your format
        self.total_samples = self.num_demos * self.steps_per_demo
        
        logging.info(f"Loaded {self.num_demos} demonstrations with {self.total_samples} total samples")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample (point_cloud, pose) pair.
        
        Args:
            idx: Global index across all demonstrations and steps
            
        Returns:
            Tuple of (point_cloud, pose) tensors
            - point_cloud: (1000, 3) tensor
            - pose: (9,) tensor
        """
        # Convert global index to demo_idx and step_idx
        demo_idx = idx // self.steps_per_demo
        step_idx = idx % self.steps_per_demo
        
        demo_key = self.demo_keys[demo_idx]
        
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_group = f[demo_key]
            
            # Load path and point cloud for this demonstration
            path = demo_group['path'][:]  # (64, 9)
            point_cloud_seq = demo_group['point_cloud'][:]  # (64, 1000, 3)
            
            # Extract the specific step
            pose = path[step_idx]  # (9,)
            point_cloud = point_cloud_seq[step_idx]  # (1000, 3)
        
        # Convert to tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        pose = torch.from_numpy(pose).float()
        
        # Normalize point cloud if requested
        if self.normalize_point_clouds:
            point_cloud = self._normalize_point_cloud(point_cloud)
        
        # Apply transforms if provided
        if self.transform:
            point_cloud = self.transform(point_cloud)
        
        return point_cloud.to(self.device), pose.to(self.device)
    
    def _normalize_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Normalize point cloud to unit sphere.
        
        Args:
            point_cloud: (N, 3) point cloud tensor
            
        Returns:
            Normalized point cloud tensor
        """
        # Center the point cloud
        centroid = torch.mean(point_cloud, dim=0, keepdim=True)
        point_cloud = point_cloud - centroid
        
        # Scale to unit sphere
        max_dist = torch.max(torch.norm(point_cloud, dim=1))
        if max_dist > 0:
            point_cloud = point_cloud / max_dist
            
        return point_cloud
    
    def get_demo_sequence(self, demo_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a complete demonstration sequence.
        
        Args:
            demo_idx: Index of the demonstration
            
        Returns:
            Tuple of (point_cloud_sequence, path) tensors
            - point_cloud_sequence: (64, 1000, 3) tensor
            - path: (64, 9) tensor
        """
        if demo_idx >= self.num_demos:
            raise IndexError(f"Demo index {demo_idx} out of range [0, {self.num_demos-1}]")
        
        demo_key = self.demo_keys[demo_idx]
        
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_group = f[demo_key]
            path = torch.from_numpy(demo_group['path'][:]).float()
            point_cloud_seq = torch.from_numpy(demo_group['point_cloud'][:]).float()
        
        # Normalize point clouds if requested
        if self.normalize_point_clouds:
            for i in range(point_cloud_seq.shape[0]):
                point_cloud_seq[i] = self._normalize_point_cloud(point_cloud_seq[i])
        
        return point_cloud_seq.to(self.device), path.to(self.device)
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Compute statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        pose_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        pc_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        
        # Sample a subset for statistics (to avoid loading everything)
        sample_size = min(1000, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        poses = []
        point_clouds = []
        
        for idx in indices:
            pc, pose = self[idx]
            poses.append(pose.cpu().numpy())
            point_clouds.append(pc.cpu().numpy())
        
        poses = np.array(poses)  # (sample_size, 9)
        point_clouds = np.array(point_clouds)  # (sample_size, 1000, 3)
        
        # Pose statistics
        pose_stats['min'] = poses.min(axis=0).tolist()
        pose_stats['max'] = poses.max(axis=0).tolist()
        pose_stats['mean'] = poses.mean(axis=0).tolist()
        pose_stats['std'] = poses.std(axis=0).tolist()
        
        # Point cloud statistics
        pc_flat = point_clouds.reshape(-1, 3)
        pc_stats['min'] = pc_flat.min(axis=0).tolist()
        pc_stats['max'] = pc_flat.max(axis=0).tolist()
        pc_stats['mean'] = pc_flat.mean(axis=0).tolist()
        pc_stats['std'] = pc_flat.std(axis=0).tolist()
        
        return {
            'num_demos': self.num_demos,
            'total_samples': self.total_samples,
            'pose_stats': pose_stats,
            'point_cloud_stats': pc_stats
        }


def create_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize_point_clouds: bool = True,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        h5_file_path: Path to H5 file
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        normalize_point_clouds: Whether to normalize point clouds
        device: Device to load tensors on
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Create full dataset
    dataset = DemonstrationDataset(
        h5_file_path=h5_file_path,
        normalize_point_clouds=normalize_point_clouds,
        device=device
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    
    logging.info(f"Created dataloaders: train={len(train_loader)*batch_size}, "
                f"val={len(val_loader)*batch_size}, test={len(test_loader)*batch_size}")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example of how to use the dataloader
    h5_file_path = "../../../data/variable_demo.h5"
    
    # Create dataset
    dataset = DemonstrationDataset(h5_file_path)
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of demonstrations: {dataset.num_demos}")
    
    # Get a single sample
    point_cloud, pose = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")  # Should be (1000, 3)
    print(f"Pose shape: {pose.shape}")  # Should be (9,)
    
    # Get dataset statistics
    stats = dataset.get_data_stats()
    print("Dataset statistics:", stats)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_file_path=h5_file_path,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test a batch
    for batch_pc, batch_pose in train_loader:
        print(f"Batch point cloud shape: {batch_pc.shape}")  # Should be (batch_size, 1000, 3)
        print(f"Batch pose shape: {batch_pose.shape}")  # Should be (batch_size, 9)
        break