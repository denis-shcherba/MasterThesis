import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import random
# Assuming src.data_handling.processing.normalize_point_cloud_to_unit_sphere exists
# For self-contained example, let's define a placeholder if it's not critical for this change

from data_handling.processing import normalize_point_cloud_to_unit_sphere


class ManipulationDataset(Dataset):
    """
    A flexible dataset for loading manipulation demonstrations from H5 files,
    supporting both sequential and static regression tasks.

    Expected H5 structure for sequential data:
    - /demo_0/path: (trajectory_length, action_dim)
    - /demo_0/points: (trajectory_length, num_points, 3)

    Expected H5 structure for regression data:
    - /demo_0/path: (action_dim,)
    - /demo_0/points: (num_points, 3)
    """

    def __init__(
        self,
        h5_file_path: str,
        is_regression: bool = False,
        sequence_length: int = 1,
        action_dim: int = 9,
        num_points: int = 1000,
        normalize_points: bool = True,
        augment_data: bool = False,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',
        random_seed: int = 42,
        observation_mode: str = 'points'
    ):
        """Initializes the dataset."""
        # --- Configuration ---
        self.h5_file_path = h5_file_path
        self.is_regression = is_regression
        self.sequence_length = sequence_length if not is_regression else 1
        self.action_dim = action_dim
        self.num_points = num_points
        self.normalize_points = normalize_points
        self.augment_data = augment_data
        self.split = split
        self.observation_mode = observation_mode

        # --- Setup ---
        self.logger = logging.getLogger(__name__)
        # REFACTOR: Use an encapsulated random number generator instead of global seeds.
        self.rng = np.random.default_rng(random_seed)
        
        # --- Data Loading and Processing ---
        self._load_demonstrations()
        
        if self.split in ['train', 'val']:
            self._create_split(train_split)
            
        self._subsample_if_needed(subsample_demos)

        self.logger.info(
            f"Dataset initialized with {len(self)} samples for split '{self.split}' "
            f"(mode: {'regression' if self.is_regression else 'sequence'})."
        )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self.is_regression:
            return len(self.demonstrations)
        
        if hasattr(self, 'valid_indices'):
            return len(self.valid_indices)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetches a sample, applies processing, and returns it as a dictionary of tensors."""
        if self.is_regression:
            demo_data = self.demonstrations[idx]
            raw_obs = demo_data['obs']

            if self.observation_mode == 'points':
                processed_obs = self._process_single_point_cloud(raw_obs)
            else:
                processed_obs = raw_obs  # depth: no point processing

            sample = {
                'observation': torch.from_numpy(processed_obs).float(),
                'action': torch.from_numpy(demo_data['path']).float(),
                'demo_id': torch.tensor(demo_data['demo_id'], dtype=torch.long)
            }
            return sample

        else:
            demo_idx, start_t = self.valid_indices[idx]
            demo_data = self.demonstrations[demo_idx]
            end_t = start_t + self.sequence_length

            if self.observation_mode == 'points':
                obs_seq = [self._process_single_point_cloud(demo_data['obs'][t]) for t in range(start_t, end_t)]
            else:
                obs_seq = [demo_data['obs'][t] for t in range(start_t, end_t)]

            obs_seq = np.stack(obs_seq, axis=0)
            action_seq = np.stack([demo_data['path'][t] for t in range(start_t, end_t)], axis=0)

            previous_action = demo_data['path'][start_t - 1].astype(np.float32) if start_t > 0 else np.zeros(self.action_dim, dtype=np.float32)

            sample = {
                'observation': torch.from_numpy(obs_seq).float(),
                'action': torch.from_numpy(action_seq).float(),
                'previous_action': torch.from_numpy(previous_action).float(),
                'demo_id': torch.tensor(demo_idx, dtype=torch.long),
                'timestep': torch.tensor(start_t, dtype=torch.long)
            }

            if self.sequence_length == 1:
                sample['observation'] = sample['observation'].squeeze(0)
                sample['action'] = sample['action'].squeeze(0)

            return sample

    
    def get_demo_info(self) -> Dict[str, any]:
        """Get information about the loaded demonstrations, adapted for the current mode."""
        info = {
            'num_demonstrations': len(self.demonstrations),
            'sequence_length': self.sequence_length,
            'action_dim': self.action_dim,
            'num_points': self.num_points,
            'split': self.split,
            'mode': 'regression' if self.is_regression else 'sequence'
        }
        
        if self.is_regression:
            info['total_samples'] = len(self.demonstrations)
        else:
            info['total_sequences'] = len(self.valid_indices)
            if self.demonstrations:
                demo_lengths = [demo['path'].shape[0] for demo in self.demonstrations]
                info.update({
                    'min_demo_length': min(demo_lengths) if demo_lengths else 0,
                    'max_demo_length': max(demo_lengths) if demo_lengths else 0,
                    'avg_demo_length': np.mean(demo_lengths) if demo_lengths else 0
                })
        return info


    def _load_demonstrations(self):
        """Load demonstrations from the H5 file, branching logic based on mode."""
        self.demonstrations: List[Dict] = []
        if not self.is_regression:
            self.valid_indices: List[tuple[int, int]] = []

        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')], key=lambda x: int(x.split('_')[1]))

                for demo_idx, demo_key in enumerate(demo_keys):
                    path_data = f[f'{demo_key}/path'][()]
                    if self.observation_mode == 'depth':
                        obs_data = f[f'{demo_key}/depth'][()]
                    elif self.observation_mode == 'points':
                        obs_data = f[f'{demo_key}/points'][()]
                    else:
                        raise ValueError(f"Unsupported observation mode: {self.observation_mode}")

                    # --- REGRESSION MODE ---
                    if self.is_regression:
                        if path_data.shape != (self.action_dim,):
                            continue
                        if self.observation_mode == 'points' and obs_data.shape != (self.num_points, 3):
                            continue
                        if self.observation_mode == 'depth' and obs_data.ndim != 2:
                            continue  # e.g. (H, W)

                    # --- SEQUENCE MODE ---
                    else:
                        if path_data.ndim != 2 or path_data.shape[1] != self.action_dim:
                            continue
                        if self.observation_mode == 'points':
                            if obs_data.shape[0] != path_data.shape[0]: continue
                            if obs_data.shape[2] != 3 or obs_data.shape[1] < self.num_points: continue
                        if self.observation_mode == 'depth':
                            if obs_data.shape[0] != path_data.shape[0]: continue  # obs: (T, H, W)

                    demo_data = {
                        'path': path_data.astype(np.float32),
                        'obs': obs_data.astype(np.float32),
                        'demo_id': demo_idx,
                        'demo_key': demo_key
                    }
                    self.demonstrations.append(demo_data)

                    if not self.is_regression:
                        num_timesteps = path_data.shape[0]
                        for t in range(num_timesteps - self.sequence_length + 1):
                            self.valid_indices.append((demo_idx, t))

        except Exception as e:
            self.logger.error(f"Failed to load H5 file {self.h5_file_path}: {e}")
            raise

        if not self.demonstrations:
            raise ValueError(f"No valid demonstrations found in {self.h5_file_path}")


    def _create_split(self, train_split: float):
        """Create train/validation split based on demonstration indices."""
        num_demos = len(self.demonstrations)
        if num_demos == 0: return

        indices = np.arange(num_demos)
        self.rng.shuffle(indices)

        split_idx = int(num_demos * train_split)
        if self.split == 'train':
            selected_ids = set(indices[:split_idx])
        else: # 'val'
            selected_ids = set(indices[split_idx:])

        if self.is_regression:
            self.demonstrations = [d for d in self.demonstrations if d['demo_id'] in selected_ids]
        else:
            self.valid_indices = [(d_idx, t) for d_idx, t in self.valid_indices if d_idx in selected_ids]

    def _subsample_if_needed(self, subsample_count: Optional[int]):
        """Subsamples the dataset if a count is provided."""
        # BUG FIX: This method now works for both modes.
        if subsample_count is None:
            return

        if self.is_regression:
            if subsample_count < len(self.demonstrations):
                self.logger.info(f"Subsampling {subsample_count} from {len(self.demonstrations)} regression samples.")
                indices = self.rng.choice(len(self.demonstrations), subsample_count, replace=False)
                self.demonstrations = [self.demonstrations[i] for i in indices]
        else:
            if hasattr(self, 'valid_indices') and subsample_count < len(self.valid_indices):
                self.logger.info(f"Subsampling {subsample_count} from {len(self.valid_indices)} sequences.")
                # Using random.sample is fine for lists of tuples.
                self.valid_indices = random.sample(self.valid_indices, subsample_count)
    
    def _process_single_point_cloud(self, points_data: np.ndarray) -> np.ndarray:
        """Applies subsampling/padding, normalization, and augmentation to a single point cloud."""
        # FIX: Re-introduced this crucial helper method.
        # 1. Subsample or Pad to ensure consistent point count
        if points_data.shape[0] > self.num_points:
            indices = self.rng.choice(points_data.shape[0], self.num_points, replace=False)
            processed_points = points_data[indices]
        elif points_data.shape[0] < self.num_points:
            pad_width = self.num_points - points_data.shape[0]
            last_point = points_data[-1] if points_data.shape[0] > 0 else np.zeros(3)
            padding = np.tile(last_point, (pad_width, 1))
            processed_points = np.concatenate([points_data, padding], axis=0)
        else:
            processed_points = points_data

        # 2. Normalize if enabled
        if self.normalize_points:
            processed_points = normalize_point_cloud_to_unit_sphere(processed_points)

        # 3. Augment if enabled
        if self.augment_data:
            processed_points = self._augment_point_cloud(processed_points)
        
        return processed_points.astype(np.float32)

    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Placeholder for point cloud augmentation logic."""
        # TODO: Implement your desired augmentations (e.g., jitter, rotation).
        return points

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
    random_seed: int = 42,
    is_regression: bool = False,
    observation_mode: str = 'points'
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
        random_seed=random_seed,
        is_regression=is_regression,
        observation_mode=observation_mode
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
        random_seed=random_seed,
        is_regression=is_regression,
        observation_mode=observation_mode
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
        num_points=data_cfg.get('num_points', 0),
        train_split=data_cfg.train_split,
        normalize_points=data_cfg.get('normalize_points', True),
        augment_data=data_cfg.get('augment_data', False),
        num_workers=data_cfg.num_workers,
        subsample_demos=data_cfg.get('subsample_demos', None),
        random_seed=data_cfg.random_seed,
        is_regression=data_cfg.get('is_regression', False),
        observation_mode=cfg.get('observation_mode', 'points')
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