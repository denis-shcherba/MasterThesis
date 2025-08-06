import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging


class ManipulationDataset(Dataset):
    """
    A flexible dataset for loading manipulation demonstrations from H5 files,
    supporting both sequential and static regression tasks with normalization.
    This version uses zero-padding for initial sequences.

    Expected H5 structure for sequential data:
    - /demo_0/path: (trajectory_length, action_dim)
    - /demo_0/points: (trajectory_length, num_points, 3)
    - /demo_0/depth: (trajectory_length, height, width)

    Expected H5 structure for regression data:
    - /demo_0/path: (action_dim,)
    - /demo_0/points: (num_points, 3)
    - /demo_0/depth: (height, width)
    """

    def __init__(
        self,
        h5_file_path: str,
        is_regression: bool = False,
        sequence_length: int = 1,
        action_dim: int = 9,
        num_points: int = 1000,
        normalize_depth: bool = True,
        normalize_actions: bool = True,
        augment_data: bool = False,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',
        random_seed: int = 42,
        observation_mode: str = 'depth',
        depth_normalization_method: str = 'minmax',     # 'minmax', 'zscore', or 'unit'
        action_normalization_method: str = 'minmax',    # 'minmax', 'zscore', or 'unit'
    ):
        """Initializes the dataset."""
        # --- Configuration ---
        self.h5_file_path = h5_file_path
        self.is_regression = is_regression
        self.sequence_length = sequence_length if not is_regression else 1
        self.action_dim = action_dim
        self.num_points = num_points
        self.normalize_depth = normalize_depth
        self.normalize_actions = normalize_actions
        self.augment_data = augment_data
        self.split = split
        self.observation_mode = observation_mode
        self.depth_normalization_method = depth_normalization_method
        self.action_normalization_method = action_normalization_method

        # --- Setup ---
        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(random_seed)
        
        # Normalization statistics
        self.depth_stats = None
        self.action_stats = None
        
        # --- Data Loading and Processing ---
        self._load_demonstrations()
        
        # Compute normalization statistics if needed
        if self.normalize_depth and self.observation_mode == 'depth':
            self._compute_depth_normalization_stats()
        if self.normalize_actions:
            self._compute_action_normalization_stats()
        
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
        """
        Fetches a sample. For sequential mode, this consists of a sequence of 
        (obs, action) pairs (padded if necessary) and a target action.
        """
        if self.is_regression:
            demo_data = self.demonstrations[idx]
            raw_obs = demo_data['obs']

            # Process observation
            if self.observation_mode == 'depth':
                processed_obs = self._process_single_depth_image(raw_obs)
            else:
                processed_obs = raw_obs

            # Process action
            action = demo_data['path']
            if self.normalize_actions and self.action_stats is not None:
                action = self._normalize_actions(action)

            sample = {
                'observation': torch.from_numpy(processed_obs).float(),
                'action': torch.from_numpy(action).float(),
                'demo_id': torch.tensor(demo_data['demo_id'], dtype=torch.long)
            }
            return sample

        # --- MODIFIED: Sequential mode with padding ---
        demo_idx, t = self.valid_indices[idx]
        demo_data = self.demonstrations[demo_idx]
        
        # `t` is the index of the last observation in the history.
        # The target action is at `t + 1`.
        target_action = demo_data['path'][t + 1]

        # Determine the start of the sequence slice from the demonstration
        start_idx = max(0, t - self.sequence_length + 1)
        
        # Get the available history slices
        obs_slice = demo_data['obs'][start_idx : t + 1]
        action_slice = demo_data['path'][start_idx : t + 1]
        timestep_slice = np.arange(start_idx, t + 1, dtype=np.int64)

        actual_seq_len = len(obs_slice)
        num_padding = self.sequence_length - actual_seq_len

        # --- Process and Normalize Slices ---
        # Process observations first to get the correct shape for padding
        if self.observation_mode == 'depth':
            processed_obs_slice = np.stack(
                [self._process_single_depth_image(obs) for obs in obs_slice], axis=0
            )
        else: # 'points'
            processed_obs_slice = obs_slice

        # Normalize actions (if enabled)
        if self.normalize_actions and self.action_stats is not None:
            action_slice = self._normalize_actions(action_slice)
            target_action = self._normalize_actions(target_action)

        # --- Create Padding ---
        # Create zero-padding for observations
        obs_padding_shape = (num_padding,) + processed_obs_slice.shape[1:]
        obs_padding = np.zeros(obs_padding_shape, dtype=processed_obs_slice.dtype)

        # Create zero-padding for actions and timesteps
        action_padding = np.zeros((num_padding, self.action_dim), dtype=action_slice.dtype)
        timestep_padding = np.zeros(num_padding, dtype=np.int64)

        # --- Combine Padding and Data ---
        obs_seq = np.concatenate([obs_padding, processed_obs_slice], axis=0)
        action_seq = np.concatenate([action_padding, action_slice], axis=0)
        timestep_seq = np.concatenate([timestep_padding, timestep_slice], axis=0)

        # --- Create Attention Mask ---
        # `True` for real data, `False` for padding. Crucial for Transformers.
        attention_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        attention_mask[-actual_seq_len:] = True

        # --- Assemble Final Sample ---
        sample = {
            'observation': torch.from_numpy(obs_seq).float(),
            'previous_actions': torch.from_numpy(action_seq).float(),
            'action': torch.from_numpy(target_action).float(),
            'timestep': torch.from_numpy(timestep_seq).long(),
            'attention_mask': torch.from_numpy(attention_mask), # NEW
            'demo_id': torch.tensor(demo_idx, dtype=torch.long)
        }

        return sample

    def _process_single_depth_image(self, depth: np.ndarray) -> np.ndarray:
        """Process a single depth image with optional normalization."""
        if self.normalize_depth and self.depth_stats is not None:
            depth = self._normalize_depth(depth)
        
        return depth

    def _compute_action_normalization_stats(self):
        """Compute normalization statistics for actions."""
        all_actions = []
        
        for demo in self.demonstrations:
            if self.is_regression:
                all_actions.append(demo['path'])
            else:
                all_actions.extend(demo['path'])
        
        if all_actions:
            all_actions = np.array(all_actions)
            if self.action_normalization_method == 'minmax':
                min_vals = np.min(all_actions, axis=0)
                max_vals = np.max(all_actions, axis=0)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1
                self.action_stats = {'method': 'minmax', 'min': min_vals, 'range': range_vals}
            elif self.action_normalization_method in ['zscore', 'unit']:
                mean = np.mean(all_actions, axis=0)
                std = np.std(all_actions, axis=0)
                std[std == 0] = 1
                self.action_stats = {'method': self.action_normalization_method, 'mean': mean, 'std': std}
        
        self.logger.info(f"Computed action normalization stats using '{self.action_normalization_method}' method")

    def _compute_depth_normalization_stats(self):
        """Compute normalization statistics for depth images."""
        all_depths = []
        
        for demo in self.demonstrations:
            data = demo['obs']
            if data.ndim == 2: # Regression
                all_depths.append(data.flatten())
            else: # Sequence
                for t in range(data.shape[0]):
                    all_depths.append(data[t].flatten())

        if all_depths:
            all_depths = np.concatenate(all_depths)
            if self.depth_normalization_method == 'minmax':
                min_val, max_val = np.min(all_depths), np.max(all_depths)
                range_val = max_val - min_val if max_val > min_val else 1
                self.depth_stats = {'method': 'minmax', 'min': min_val, 'range': range_val}
            elif self.depth_normalization_method in ['zscore', 'unit']:
                mean, std = np.mean(all_depths), np.std(all_depths)
                if std == 0: std = 1
                self.depth_stats = {'method': self.depth_normalization_method, 'mean': mean, 'std': std}
                
        self.logger.info(f"Computed depth normalization stats using '{self.depth_normalization_method}' method")

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if not self.normalize_actions or self.action_stats is None: return actions
        if self.action_stats['method'] == 'minmax':
            return (actions - self.action_stats['min']) / self.action_stats['range']
        elif self.action_stats['method'] in ['zscore', 'unit']:
            return (actions - self.action_stats['mean']) / self.action_stats['std']
        return actions

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        if not self.normalize_depth or self.depth_stats is None: return depth
        if self.depth_stats['method'] == 'minmax':
            return (depth - self.depth_stats['min']) / self.depth_stats['range']
        elif self.depth_stats['method'] in ['zscore', 'unit']:
            return (depth - self.depth_stats['mean']) / self.depth_stats['std']
        return depth
        
    def get_normalization_stats(self) -> Dict:
        """Get the computed normalization statistics."""
        return {'depth_stats': self.depth_stats, 'action_stats': self.action_stats}

    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        if not self.normalize_actions or self.action_stats is None: return normalized_actions
        if self.action_stats['method'] == 'minmax':
            return normalized_actions * self.action_stats['range'] + self.action_stats['min']
        elif self.action_stats['method'] in ['zscore', 'unit']:
            return normalized_actions * self.action_stats['std'] + self.action_stats['mean']
        return normalized_actions
        
    def get_demo_info(self) -> Dict[str, any]:
        """Get information about the loaded demonstrations."""
        info = {
            'num_demonstrations': len(self.demonstrations),
            'sequence_length': self.sequence_length,
            'action_dim': self.action_dim,
            'num_points': self.num_points,
            'split': self.split,
            'mode': 'regression' if self.is_regression else 'sequence',
            'observation_mode': self.observation_mode,
            'normalize_depth': self.normalize_depth,
            'normalize_actions': self.normalize_actions
        }
        
        if self.is_regression:
            info['total_samples'] = len(self.demonstrations)
        else:
            info['total_predictable_steps'] = len(self.valid_indices)
            if self.demonstrations:
                demo_lengths = [demo['path'].shape[0] for demo in self.demonstrations]
                info.update({
                    'min_demo_length': min(demo_lengths) if demo_lengths else 0,
                    'max_demo_length': max(demo_lengths) if demo_lengths else 0,
                    'avg_demo_length': np.mean(demo_lengths) if demo_lengths else 0
                })
        return info

    def _load_demonstrations(self):
        """Load demonstrations from the H5 file."""
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

                    if self.is_regression:
                        if path_data.shape != (self.action_dim,): continue
                    else:
                        if path_data.ndim != 2 or path_data.shape[1] != self.action_dim: continue
                        if obs_data.shape[0] != path_data.shape[0]: continue

                    demo_data = {
                        'path': path_data.astype(np.float32),
                        'obs': obs_data.astype(np.float32),
                        'demo_id': demo_idx,
                        'demo_key': demo_key
                    }
                    self.demonstrations.append(demo_data)

                    # --- MODIFIED: Indexing logic ---
                    if not self.is_regression:
                        num_timesteps = path_data.shape[0]
                        # Create a valid index for every possible step in the trajectory
                        # where a next action can be predicted.
                        # `t` represents the index of the *last observation* in the history.
                        # The goal is to predict the action at `t+1`.
                        for t in range(num_timesteps - 1):
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
        selected_ids = set(indices[:split_idx] if self.split == 'train' else indices[split_idx:])

        if self.is_regression:
            self.demonstrations = [d for d in self.demonstrations if d['demo_id'] in selected_ids]
        else:
            # We need to filter demonstrations first, then re-index `valid_indices`
            original_demonstrations = self.demonstrations
            self.demonstrations = [d for d in original_demonstrations if d['demo_id'] in selected_ids]
            
            # Create a map from old demo_id to new demo_id
            demo_id_map = {d['demo_id']: new_id for new_id, d in enumerate(self.demonstrations)}
            
            # Rebuild valid_indices based on the filtered and re-indexed demonstrations
            new_valid_indices = []
            for old_demo_idx, t in self.valid_indices:
                if old_demo_idx in selected_ids:
                    new_demo_idx = demo_id_map[old_demo_idx]
                    new_valid_indices.append((new_demo_idx, t))
            self.valid_indices = new_valid_indices

    def _subsample_if_needed(self, subsample_demos: Optional[int]):
        """Subsample demonstrations if requested."""
        if subsample_demos is not None and subsample_demos > 0:
            if len(self.demonstrations) > subsample_demos:
                # This logic works for both modes because splitting/subsampling now filters
                # `self.demonstrations` and rebuilds `valid_indices` accordingly.
                self.demonstrations = self.demonstrations[:subsample_demos]
                kept_demo_ids = {d['demo_id'] for d in self.demonstrations}
                
                if not self.is_regression:
                    self.valid_indices = [(d_idx, t) for d_idx, t in self.valid_indices if d_idx in kept_demo_ids]

def create_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    action_dim: int = 9,
    num_points: int = 1000,
    train_split: float = 0.8,
    augment_data: bool = True,
    num_workers: int = 4,
    subsample_demos: Optional[int] = None,
    random_seed: int = 42,
    is_regression: bool = False,
    observation_mode: str = 'points',
    normalize_depth: bool = True,
    normalize_actions: bool = True,
    depth_normalization_method = 'minmax',
    action_normalization_method = 'zscore'

) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        augment_data=augment_data,
        subsample_demos=subsample_demos,
        train_split=train_split,
        split='train',
        random_seed=random_seed,
        is_regression=is_regression,
        observation_mode=observation_mode,
        normalize_depth=normalize_depth,
        normalize_actions=normalize_actions,
        depth_normalization_method=depth_normalization_method,
        action_normalization_method=action_normalization_method
    )

    val_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        augment_data=False,
        subsample_demos=None,
        train_split=train_split,
        split='val',
        random_seed=random_seed,
        is_regression=is_regression,
        observation_mode=observation_mode,
        normalize_depth=normalize_depth,
        normalize_actions=normalize_actions,
        depth_normalization_method=depth_normalization_method,
        action_normalization_method=action_normalization_method
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
        normalize_depth=data_cfg.get('normalize_depth', True),
        normalize_actions=data_cfg.get('normalize_actions', True),
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