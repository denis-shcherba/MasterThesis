import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import random
# Assuming src.data_handling.processing.normalize_point_cloud_to_unit_sphere exists
# For self-contained example, let's define a placeholder if it's not critical for this change

from data_handling.processing import normalize_point_cloud_to_unit_sphere


class ManipulationDataset(Dataset):
    """
    Dataset for loading manipulation demonstration data from H5 files.

    Data structure expected:
    - /demo_0/path: (trajectory_length, action_dim) array - trajectory information
    - /demo_0/points: (trajectory_length, num_points, 3) array - point cloud data
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

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Ensure logger messages are displayed

        self._load_demonstrations()

        if split != 'all':
            self._create_split(train_split, random_seed)

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
                    path_data = f[f'{demo_key}/path'][()]
                    points_data = f[f'{demo_key}/points'][()]

                    if path_data.shape[0] != points_data.shape[0]:
                        self.logger.warning(f"Mismatched timesteps in {demo_key}: "
                                          f"path {path_data.shape[0]}, points {points_data.shape[0]}")
                        continue

                    if path_data.shape[1] != self.action_dim:
                        self.logger.warning(f"Action dimension mismatch in {demo_key}: "
                                          f"expected {self.action_dim}, got {path_data.shape[1]}")
                        continue

                    # Allowing for flexibility in num_points if it's subsampled later,
                    # but the H5 should ideally match or be superset.
                    if points_data.shape[2] != 3 or points_data.shape[1] < self.num_points :
                         self.logger.warning(f"Point cloud shape mismatch or insufficient points in {demo_key}: "
                                           f"expected at least ({self.num_points}, 3), got {points_data.shape[1:]}. "
                                           f"Ensure point clouds have at least 'num_points' points.")
                         # If you want to strictly enforce num_points from H5, uncomment below
                         # if points_data.shape[1:] != (self.num_points, 3):
                         #    continue


                    demo_data = {
                        'path': path_data.astype(np.float32),
                        'points': points_data.astype(np.float32),
                        'demo_id': demo_idx,
                        'demo_key': demo_key
                    }
                    self.demonstrations.append(demo_data)

                    num_timesteps = path_data.shape[0]
                    # A sequence is valid if it has sequence_length steps and a next_action can be determined
                    # For previous_action, t=0 is handled with zeros.
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
        num_demos = len(self.demonstrations)
        demo_indices = list(range(num_demos))

        rng = np.random.RandomState(random_seed)
        rng.shuffle(demo_indices)

        split_idx = int(num_demos * train_split)
        if self.split == 'train':
            selected_demos = set(demo_indices[:split_idx])
        elif self.split == 'val':
            selected_demos = set(demo_indices[split_idx:])
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.valid_indices = [
            (demo_idx, t) for demo_idx, t in self.valid_indices
            if demo_idx in selected_demos
        ]

    # TODO, maybe drop
    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud."""
        if not self.augment_data:
            return points

        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points_rotated = points @ rotation_matrix.T

        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, points_rotated.shape).astype(np.float32)
        points_augmented = points_rotated + noise

        if np.random.random() < 0.3:
            num_dropout = int(0.1 * len(points_augmented))
            if len(points_augmented) - num_dropout > 0 : # ensure we don't drop all points
                keep_indices = np.random.choice(
                    len(points_augmented),
                    len(points_augmented) - num_dropout,
                    replace=False
                )
                if len(keep_indices) > 0: # Ensure there are points to duplicate from
                    duplicate_indices = np.random.choice(keep_indices, num_dropout) # This can be problematic if keep_indices is small
                    final_indices = np.concatenate([keep_indices, duplicate_indices])
                    points_augmented = points_augmented[final_indices]


        return points_augmented

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'point_cloud': Point cloud tensor (sequence_length, num_points, 3) or (num_points, 3) if sequence_length=1
                - 'action': Action tensor (sequence_length, action_dim) or (action_dim) if sequence_length=1
                - 'previous_action': Previous action tensor (action_dim) - state input
                - 'next_action': Next action tensor (action_dim) - for next step prediction
                - 'demo_id': Demonstration ID
                - 'timestep': Starting timestep in the demonstration
        """
        demo_idx, start_t = self.valid_indices[idx]
        demo_data = self.demonstrations[demo_idx]

        end_t = start_t + self.sequence_length

        point_clouds = []
        actions = []

        for t in range(start_t, end_t):
            # Subsample points if necessary (e.g., H5 has more points than self.num_points)
            current_points_data = demo_data['points'][t]
            if current_points_data.shape[0] > self.num_points:
                # Randomly subsample points
                indices = np.random.choice(current_points_data.shape[0], self.num_points, replace=False)
                points = current_points_data[indices].copy()
            elif current_points_data.shape[0] < self.num_points:
                # Pad with last point or zeros if fewer points than num_points (or raise error)
                # This case should ideally be handled by data preparation or an error in _load_demonstrations
                self.logger.warning(f"Demo {demo_idx}, timestep {t} has fewer points ({current_points_data.shape[0]}) than required ({self.num_points}). Padding with last point.")
                points = np.zeros((self.num_points, 3), dtype=np.float32)
                points[:current_points_data.shape[0], :] = current_points_data
                if current_points_data.shape[0] > 0 : # pad with last point
                    points[current_points_data.shape[0]:, :] = current_points_data[-1]

            else:
                points = current_points_data.copy()


            if self.normalize_points:
                # Using the imported normalize_point_cloud_to_unit_sphere function first
                points = normalize_point_cloud_to_unit_sphere(points)


            points = self._augment_point_cloud(points)

            point_clouds.append(points)
            actions.append(demo_data['path'][t])

        point_cloud_seq = np.stack(point_clouds, axis=0)
        action_seq = np.stack(actions, axis=0)

        # Determine previous action (state)
        if start_t == 0:
            # For the first timestep in the demonstration, use a zero vector
            previous_action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            previous_action = demo_data['path'][start_t - 1].astype(np.float32)

        if end_t < demo_data['path'].shape[0]:
            next_action = demo_data['path'][end_t].astype(np.float32)
        else:
            next_action = demo_data['path'][-1].astype(np.float32) # Use last action if at end

        sample = {
            'point_cloud': torch.from_numpy(point_cloud_seq),
            'action': torch.from_numpy(action_seq),
            'previous_action': torch.from_numpy(previous_action),
            'next_action': torch.from_numpy(next_action),
            'demo_id': torch.tensor(demo_idx, dtype=torch.long),
            'timestep': torch.tensor(start_t, dtype=torch.long)
        }

        if self.sequence_length == 1:
            sample['point_cloud'] = sample['point_cloud'].squeeze(0)
            sample['action'] = sample['action'].squeeze(0)
            # 'previous_action' is already (action_dim), so no squeeze needed for seq dim

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
                'min_demo_length': min(demo_lengths) if demo_lengths else 0,
                'max_demo_length': max(demo_lengths) if demo_lengths else 0,
                'avg_demo_length': np.mean(demo_lengths) if demo_lengths else 0
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
        augment_data=False,
        subsample_demos=None,
        train_split=train_split,
        split='val',
        random_seed=random_seed
    )

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


def create_dataloaders_from_config(cfg) -> Tuple[DataLoader, DataLoader]:
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
    from omegaconf import DictConfig, OmegaConf
    import os

    # Create a dummy config for testing if not running with Hydra
    # A minimal config structure for the data section.
    dummy_config_yaml = """
data:
  h5_file_path: "dummy_data.h5" # Will be created if it doesn't exist
  batch_size: 4
  sequence_length: 1 # Test with 1
  action_dim: 9
  num_points: 1000
  train_split: 0.8
  normalize_points: true
  augment_data: true
  num_workers: 0 # Set to 0 for easier debugging in main thread
  subsample_demos: null
  random_seed: 42

# Minimal hydra config for standalone execution
hydra:
  run:
    dir: ./outputs_test_dataloader/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: ./multirun_test_dataloader/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
"""
    # This is a simplified main for testing, assuming Hydra might not be fully set up.
    # Try to use Hydra if available, otherwise use a dummy config.
    try:
        # Ensure the config path is relative to this file if needed, or use absolute paths
        # For robust testing, Hydra's compose API is better than relying on @hydra.main decorator finding configs
        # This example assumes `../../configs` and `config.yaml` might not exist in this context,
        # so it falls back to a dummy config.

        # Create dummy H5 file for testing
        print("Creating dummy test data for 'dummy_data.h5'...")
        dummy_h5_path = "dummy_data.h5"
        num_dummy_demos = 5
        demo_length = 64 # timesteps per demo
        action_dim_dummy = 9
        num_points_dummy = 1000

        with h5py.File(dummy_h5_path, 'w') as f:
            for i in range(num_dummy_demos):
                demo_group = f.create_group(f'demo_{i}')
                demo_group.create_dataset('path', data=np.random.randn(demo_length, action_dim_dummy).astype(np.float32))
                demo_group.create_dataset('points', data=np.random.randn(demo_length, num_points_dummy, 3).astype(np.float32))
        print(f"Dummy data created at {dummy_h5_path}")

        cfg = OmegaConf.create(dummy_config_yaml) # Load dummy config
        data_cfg = cfg.data

        print(f"Using data config: {OmegaConf.to_yaml(data_cfg)}")

        # Test with sequence_length = 1
        print("\n--- Testing with sequence_length = 1 ---")
        train_loader_seq1, val_loader_seq1 = create_dataloaders(
            h5_file_path=data_cfg.h5_file_path,
            batch_size=data_cfg.batch_size,
            sequence_length=1, # Explicitly set for this test
            action_dim=data_cfg.action_dim,
            num_points=data_cfg.num_points,
            train_split=data_cfg.train_split,
            normalize_points=data_cfg.normalize_points,
            augment_data=data_cfg.augment_data,
            num_workers=data_cfg.num_workers,
            subsample_demos=data_cfg.get('subsample_demos', None),
            random_seed=data_cfg.random_seed
        )

        print(f"Train samples (seq_len=1): {len(train_loader_seq1.dataset)}")
        print(f"Val samples (seq_len=1): {len(val_loader_seq1.dataset)}")

        for i, batch in enumerate(train_loader_seq1):
            print(f"\nBatch {i+1} shapes (seq_len=1):")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
            if i == 0: break # Only show first batch

        # Test with sequence_length = 5
        print("\n--- Testing with sequence_length = 5 ---")
        # Ensure demos are long enough for sequence_length 5
        # Valid indices are num_timesteps - sequence_length + 1
        # 64 - 5 + 1 = 60 sequences per demo
        train_loader_seq5, val_loader_seq5 = create_dataloaders(
            h5_file_path=data_cfg.h5_file_path,
            batch_size=data_cfg.batch_size,
            sequence_length=5, # Explicitly set for this test
            action_dim=data_cfg.action_dim,
            num_points=data_cfg.num_points,
            train_split=data_cfg.train_split,
            normalize_points=data_cfg.normalize_points,
            augment_data=data_cfg.augment_data,
            num_workers=data_cfg.num_workers,
            subsample_demos=data_cfg.get('subsample_demos', None),
            random_seed=data_cfg.random_seed
        )
        print(f"Train samples (seq_len=5): {len(train_loader_seq5.dataset)}")
        print(f"Val samples (seq_len=5): {len(val_loader_seq5.dataset)}")

        for i, batch in enumerate(train_loader_seq5):
            print(f"\nBatch {i+1} shapes (seq_len=5):")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
            # Check a specific previous_action from a non-first step if possible
            demo_ids = batch['demo_id']
            timesteps = batch['timestep']
            # Find a sample in the batch that is not t=0
            non_zero_t_idx = (timesteps > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_t_idx) > 0:
                sample_idx_in_batch = non_zero_t_idx[0].item()
                demo_id_val = demo_ids[sample_idx_in_batch].item()
                timestep_val = timesteps[sample_idx_in_batch].item()
                prev_action_sample = batch['previous_action'][sample_idx_in_batch]

                # Retrieve the ground truth previous action from H5 to verify
                with h5py.File(data_cfg.h5_file_path, 'r') as f_verify:
                    # Find the original demo key if needed, but demo_id maps to self.demonstrations
                    # For simplicity, assume demo_id maps directly to the order in self.demonstrations
                    # This requires knowing which demo_key corresponds to demo_id_val
                    # In this test setup, demo_idx in _load_demonstrations becomes the demo_id.
                    original_demo_data = train_loader_seq5.dataset.demonstrations[demo_id_val]['path']
                    ground_truth_prev_action = original_demo_data[timestep_val -1]
                    #print(f"  Sample (demo_id={demo_id_val}, timestep={timestep_val}):")
                    #print(f"    Returned previous_action: {prev_action_sample.numpy()}")
                    #print(f"    Ground_truth previous_action from H5: {ground_truth_prev_action}")
                    assert np.allclose(prev_action_sample.numpy(), ground_truth_prev_action), "Previous action mismatch!"
                    print(f"    Previous action for demo {demo_id_val}, t={timestep_val} successfully verified.")

            if i == 0: break # Only show first batch

        # Clean up dummy file
        if os.path.exists(dummy_h5_path):
            os.remove(dummy_h5_path)
            print(f"\nCleaned up {dummy_h5_path}")

    except Exception as e:
        print(f"Error in example usage: {e}")
        import traceback
        traceback.print_exc()