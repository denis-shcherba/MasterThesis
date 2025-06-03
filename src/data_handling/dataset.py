import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import random
from src.data_handling.processing import normalize_point_cloud_to_unit_sphere 

class ManipulationDataset(Dataset):
    """
    Dataset for loading manipulation demonstration data from H5 files.
    
    Data structure expected:
    - /demo_0/path: (64, 9) array - trajectory information
    - /demo_0/points: (64, 1000, 3) array - point cloud data
    """
    
    def __init__(
        self,
        h5_file_path: str,
        sequence_length: int = 1,
        action_dim: int = 9,
        num_points: int = 1000,
        normalize_points: bool = True,
        augment_data: bool = False,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',  # 'train', 'val', or 'all'
        random_seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            h5_file_path: Path to the H5 file containing demonstrations
            sequence_length: Number of consecutive timesteps to use as input
            action_dim: Dimension of action space (should match path dimension)
            num_points: Number of points in each point cloud
            normalize_points: Whether to normalize point clouds
            augment_data: Whether to apply data augmentation
            subsample_demos: If specified, randomly subsample this many demos
            train_split: Fraction of data to use for training
            split: Which split to use ('train', 'val', or 'all')
            random_seed: Random seed for reproducibility
        """
        self.h5_file_path = h5_file_path
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.num_points = num_points
        self.normalize_points = normalize_points
        self.augment_data = augment_data
        self.split = split
        
        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.logger = logging.getLogger(__name__)
        
        # Load and process data
        self._load_demonstrations()
        
        # Create train/val split
        if split != 'all':
            self._create_split(train_split, random_seed)
        
        # Subsample demonstrations if requested
        if subsample_demos is not None and subsample_demos < len(self.valid_indices):
            self.valid_indices = random.sample(self.valid_indices, subsample_demos)
        
        self.logger.info(f"Dataset initialized with {len(self)} samples ({split} split)")
    
    def _load_demonstrations(self):
        """Load demonstrations from H5 file."""
        self.demonstrations = []
        self.valid_indices = []
        
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_keys = [key for key in f.keys() if key.startswith('demo_')]
            demo_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            for demo_idx, demo_key in enumerate(demo_keys):
                try:
                    # Load path and points data
                    path_data = f[f'{demo_key}/path'][()]  # (64, 9)
                    points_data = f[f'{demo_key}/points'][()]  # (64, 1000, 3)

                    # Validate data shapes
                    if path_data.shape[0] != points_data.shape[0]:
                        self.logger.warning(f"Mismatched timesteps in {demo_key}: "
                                          f"path {path_data.shape[0]}, points {points_data.shape[0]}")
                        continue
                    
                    if path_data.shape[1] != self.action_dim:
                        self.logger.warning(f"Action dimension mismatch in {demo_key}: "
                                          f"expected {self.action_dim}, got {path_data.shape[1]}")
                        continue
                    
                    if points_data.shape[1:] != (self.num_points, 3):
                        self.logger.warning(f"Point cloud shape mismatch in {demo_key}: "
                                          f"expected ({self.num_points}, 3), got {points_data.shape[1:]}")
                        continue
                    
                    # Store demonstration data
                    demo_data = {
                        'path': path_data.astype(np.float32),
                        'points': points_data.astype(np.float32),
                        'demo_id': demo_idx,
                        'demo_key': demo_key
                    }
                    
                    self.demonstrations.append(demo_data)
                    
                    # Create valid indices for each timestep sequence
                    num_timesteps = path_data.shape[0]
                    for t in range(num_timesteps - self.sequence_length + 1):
                        self.valid_indices.append((demo_idx, t))
                
                except Exception as e:
                    self.logger.error(f"Error loading {demo_key}: {e}")
                    continue
        
        if not self.demonstrations:
            raise ValueError(f"No valid demonstrations found in {self.h5_file_path}")
        
        self.logger.info(f"Loaded {len(self.demonstrations)} demonstrations "
                        f"with {len(self.valid_indices)} total sequences")
    
    def _create_split(self, train_split: float, random_seed: int):
        """Create train/validation split."""
        # Split by demonstrations (not individual timesteps)
        num_demos = len(self.demonstrations)
        demo_indices = list(range(num_demos))
        
        # Shuffle demonstrations
        rng = np.random.RandomState(random_seed)
        rng.shuffle(demo_indices)
        
        # Split demonstrations
        split_idx = int(num_demos * train_split)
        if self.split == 'train':
            selected_demos = set(demo_indices[:split_idx])
        elif self.split == 'val':
            selected_demos = set(demo_indices[split_idx:])
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Filter valid indices based on selected demonstrations
        self.valid_indices = [
            (demo_idx, t) for demo_idx, t in self.valid_indices
            if demo_idx in selected_demos
        ]
    
    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud to unit sphere.
        
        Args:
            points: Point cloud array of shape (num_points, 3)
            
        Returns:
            Normalized point cloud
        """
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 1e-8:  # Avoid division by zero
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        
        return points_normalized.astype(np.float32)
    
    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to point cloud.
        
        Args:
            points: Point cloud array of shape (num_points, 3)
            
        Returns:
            Augmented point cloud
        """
        if not self.augment_data:
            return points
        
        # Random rotation around z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        points_rotated = points @ rotation_matrix.T
        
        # Add small amount of noise
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, points_rotated.shape).astype(np.float32)
        points_augmented = points_rotated + noise
        
        # Random point dropout (remove some points and duplicate others)
        if np.random.random() < 0.3:  # 30% chance of dropout
            num_dropout = int(0.1 * len(points_augmented))  # Drop 10% of points
            keep_indices = np.random.choice(
                len(points_augmented), 
                len(points_augmented) - num_dropout, 
                replace=False
            )
            duplicate_indices = np.random.choice(keep_indices, num_dropout)
            final_indices = np.concatenate([keep_indices, duplicate_indices])
            points_augmented = points_augmented[final_indices]
        
        return points_augmented
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'point_cloud': Point cloud tensor (sequence_length, num_points, 3)
                - 'action': Action tensor (sequence_length, action_dim)
                - 'next_action': Next action tensor (action_dim) - for next step prediction
                - 'demo_id': Demonstration ID
                - 'timestep': Starting timestep in the demonstration
        """
        demo_idx, start_t = self.valid_indices[idx]
        demo_data = self.demonstrations[demo_idx]
        
        # Extract sequences
        end_t = start_t + self.sequence_length
        
        point_clouds = []
        actions = []
        
        for t in range(start_t, end_t):
            # Get point cloud
            points = demo_data['points'][t].copy()
            
            # Apply normalization
            if self.normalize_points:
                
                points = normalize_point_cloud_to_unit_sphere(points) # Use the utility function
                points = self._normalize_point_cloud(points)
            
            # Apply augmentation
            points = self._augment_point_cloud(points)
            
            point_clouds.append(points)
            actions.append(demo_data['path'][t])
        
        # Stack sequences
        point_cloud_seq = np.stack(point_clouds, axis=0)  # (seq_len, num_points, 3)
        action_seq = np.stack(actions, axis=0)  # (seq_len, action_dim)
        
        # Get next action for prediction (if available)
        if end_t < demo_data['path'].shape[0]:
            next_action = demo_data['path'][end_t]
        else:
            next_action = demo_data['path'][-1]  # Use last action if at end
        
        # Convert to tensors
        sample = {
            'point_cloud': torch.from_numpy(point_cloud_seq),
            'action': torch.from_numpy(action_seq),
            'next_action': torch.from_numpy(next_action.astype(np.float32)),
            'demo_id': torch.tensor(demo_idx, dtype=torch.long),
            'timestep': torch.tensor(start_t, dtype=torch.long)
        }
        
        # If sequence length is 1, squeeze the sequence dimension
        if self.sequence_length == 1:
            sample['point_cloud'] = sample['point_cloud'].squeeze(0)
            sample['action'] = sample['action'].squeeze(0)
        
        return sample
    
    def get_demo_info(self) -> Dict[str, any]:
        """Get information about the loaded demonstrations."""
        info = {
            'num_demonstrations': len(self.demonstrations),
            'total_sequences': len(self.valid_indices),
            'sequence_length': self.sequence_length,
            'action_dim': self.action_dim,
            'num_points': self.num_points,
            'split': self.split
        }
        
        if self.demonstrations:
            demo_lengths = [demo['path'].shape[0] for demo in self.demonstrations]
            info.update({
                'min_demo_length': min(demo_lengths),
                'max_demo_length': max(demo_lengths),
                'avg_demo_length': np.mean(demo_lengths)
            })
        
        return info


def create_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    action_dim: int = 9,
    num_points: int = 1000,
    train_split: float = 0.8,
    normalize_points: bool = True,
    augment_data: bool = True,
    num_workers: int = 4,
    subsample_demos: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        h5_file_path: Path to H5 file with demonstrations
        batch_size: Batch size for dataloaders
        sequence_length: Length of input sequences
        action_dim: Dimension of action space
        num_points: Number of points per point cloud
        train_split: Fraction of data for training
        normalize_points: Whether to normalize point clouds
        augment_data: Whether to apply data augmentation (only for training)
        num_workers: Number of worker processes for data loading
        subsample_demos: Optional number of demos to subsample
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        normalize_points=normalize_points,
        augment_data=augment_data,
        subsample_demos=subsample_demos,
        train_split=train_split,
        split='train',
        random_seed=random_seed
    )
    
    val_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        normalize_points=normalize_points,
        augment_data=False,  # No augmentation for validation
        subsample_demos=None,  # Use all validation data
        train_split=train_split,
        split='val',
        random_seed=random_seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples with variable sequence lengths.
    """
    # This is the default behavior, but you can customize if needed
    return torch.utils.data.dataloader.default_collate(batch)



def create_dataloaders_from_config(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders from Hydra config.
    
    Args:
        cfg: Hydra config object with data section
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = cfg.data
    return create_dataloaders(
        h5_file_path=data_cfg.h5_file_path,
        batch_size=data_cfg.batch_size,
        sequence_length=data_cfg.sequence_length,
        action_dim=data_cfg.action_dim,
        num_points=data_cfg.num_points,
        train_split=data_cfg.train_split,
        normalize_points=data_cfg.normalize_points,
        augment_data=data_cfg.augment_data,
        num_workers=data_cfg.num_workers,
        subsample_demos=data_cfg.get('subsample_demos', None),
        random_seed=data_cfg.random_seed
    )


# Example usage and testing
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_path="../../configs", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        # Extract data config
        data_cfg = cfg.data

        try:
            # Create dataloaders using config values
            train_loader, val_loader = create_dataloaders(
                h5_file_path=data_cfg.h5_file_path,
                batch_size=data_cfg.batch_size,
                sequence_length=data_cfg.sequence_length,
                action_dim=data_cfg.action_dim,
                num_points=data_cfg.num_points,
                train_split=data_cfg.train_split,
                normalize_points=data_cfg.normalize_points,
                augment_data=data_cfg.augment_data,
                num_workers=data_cfg.num_workers,
                subsample_demos=data_cfg.get('subsample_demos', None),
                random_seed=data_cfg.random_seed
            )
            
            print("Dataset created successfully!")
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            
            # Test loading a batch
            for batch in train_loader:
                print(f"Batch shapes:")
                for key, value in batch.items():
                    print(f"  {key}: {value.shape}")
                break
            
        except Exception as e:
            print(f"Error testing dataset: {e}")
            # Create a dummy dataset for testing
            print("Creating dummy test data...")
            
            import h5py
            dummy_path = "dummy_data.h5"
            
            with h5py.File(dummy_path, 'w') as f:
                for i in range(5):
                    demo_group = f.create_group(f'demo_{i}')
                    demo_group.create_dataset('path', data=np.random.randn(64, 9))
                    demo_group.create_dataset('points', data=np.random.randn(64, 1000, 3))
            
            # Test with dummy data
            train_loader, val_loader = create_dataloaders(
                h5_file_path=dummy_path,
                batch_size=4,
                num_workers=0
            )
            
            print(f"Dummy dataset - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
            
            for batch in train_loader:
                print("Dummy batch loaded successfully!")
                for key, value in batch.items():
                    print(f"  {key}: {value.shape}")
                break

    # Call the main function
    main()