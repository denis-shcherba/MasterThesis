import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging


class ManipulationDataset(Dataset):
    """
    Streaming HDF5 dataset for imitation learning.
    Loads only the data slice needed for each __getitem__ call.
    """

    def __init__(
        self,
        h5_file_path: str,
        is_regression: bool = False,
        is_waypointPlusTimings: bool = False,
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
        self.h5_file_path = h5_file_path
        self.is_regression = is_regression
        self.is_waypointPlusTimings = is_waypointPlusTimings
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

        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(random_seed)

        # We'll store only dataset metadata, not the arrays
        self.demo_meta: List[Dict] = []
        self.valid_indices: List[Tuple[int, int]] = []

        # Load just the metadata first
        self._index_demonstrations()

        # Create train/val split
        if self.split in ['train', 'val']:
            self._create_split(train_split)

        # Optional subsampling
        self._subsample_if_needed(subsample_demos)

        # Compute normalization stats by streaming through the file
        with h5py.File(self.h5_file_path, 'r') as f:
            if self.normalize_actions:
                self.action_stats = self._compute_action_normalization_stats(f)
            else:
                self.action_stats = None

            if self.normalize_depth and self.observation_mode == 'depth':
                self.depth_stats = self._compute_depth_normalization_stats(f)
            else:
                self.depth_stats = None

    def _index_demonstrations(self):
        """Read only the shapes and keys from the file, no data."""
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')],
                            key=lambda x: int(x.split('_')[1]))
            for demo_idx, demo_key in enumerate(demo_keys):
                path_shape = f[f'{demo_key}/path'].shape
                if self.observation_mode == 'depth':
                    obs_shape = f[f'{demo_key}/depth'].shape
                else:
                    obs_shape = f[f'{demo_key}/points'].shape

                if self.is_regression:
                    if path_shape != (self.action_dim,):
                        continue
                else:
                    if len(path_shape) != 2 or path_shape[1] != self.action_dim:
                        continue
                    if obs_shape[0] != path_shape[0]:
                        continue

                # Build metadata dictionary once
                meta = {
                    'demo_id': demo_idx,
                    'demo_key': demo_key,
                    'path_shape': path_shape,
                    'obs_shape': obs_shape
                }

                # Extend conditionally
                if self.is_waypointPlusTimings:
                    meta['num_waypoints'] = f[f'{demo_key}/ways'].shape

                self.demo_meta.append(meta)

                if not self.is_regression:
                    num_timesteps = path_shape[0]
                    for t in range(num_timesteps - 1):
                        self.valid_indices.append((demo_idx, t))

        if not self.demo_meta:
            raise ValueError(f"No valid demonstrations found in {self.h5_file_path}")

    def _create_split(self, train_split: float):
        """Filter demos for train/val split."""
        indices = np.arange(len(self.demo_meta))
        self.rng.shuffle(indices)
        split_idx = int(len(self.demo_meta) * train_split)
        selected_ids = set(indices[:split_idx] if self.split == 'train' else indices[split_idx:])

        self.demo_meta = [d for i, d in enumerate(self.demo_meta) if i in selected_ids]
        if not self.is_regression:
            self.valid_indices = [(d_idx, t) for d_idx, t in self.valid_indices if d_idx in selected_ids]

        # Remap demo_id to sequential
        old_to_new = {old['demo_id']: i for i, old in enumerate(self.demo_meta)}
        for d in self.demo_meta:
            d['demo_id'] = old_to_new[d['demo_id']]
        if not self.is_regression:
            self.valid_indices = [(old_to_new[d_idx], t) for d_idx, t in self.valid_indices]

    def _subsample_if_needed(self, subsample_demos: Optional[int]):
        if subsample_demos is not None and subsample_demos > 0:
            if len(self.demo_meta) > subsample_demos:
                self.demo_meta = self.demo_meta[:subsample_demos]
                kept_ids = {d['demo_id'] for d in self.demo_meta}
                if not self.is_regression:
                    self.valid_indices = [(d_idx, t) for d_idx, t in self.valid_indices if d_idx in kept_ids]

    def _compute_action_normalization_stats(self, f):
        min_vals = np.full(self.action_dim, np.inf, dtype=np.float64)
        max_vals = np.full(self.action_dim, -np.inf, dtype=np.float64)
        sum_vals = np.zeros(self.action_dim, dtype=np.float64)
        sum_sq_vals = np.zeros(self.action_dim, dtype=np.float64)
        count = 0

        for meta in self.demo_meta:
            arr = f[f"{meta['demo_key']}/path"][...]
            if not self.is_regression:
                count += arr.shape[0]
                min_vals = np.minimum(min_vals, arr.min(axis=0))
                max_vals = np.maximum(max_vals, arr.max(axis=0))
                sum_vals += arr.sum(axis=0)
                sum_sq_vals += (arr ** 2).sum(axis=0)
            else:
                count += 1
                min_vals = np.minimum(min_vals, arr)
                max_vals = np.maximum(max_vals, arr)
                sum_vals += arr
                sum_sq_vals += arr ** 2

        if self.action_normalization_method == 'minmax':
            range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
            return {'method': 'minmax', 'min': min_vals, 'range': range_vals}
        else:
            mean = sum_vals / count
            var = (sum_sq_vals / count) - mean ** 2
            std = np.where(var == 0, 1, np.sqrt(var))
            return {'method': self.action_normalization_method, 'mean': mean, 'std': std}

    def _compute_depth_normalization_stats(self, f):
        min_val, max_val = np.inf, -np.inf
        sum_val, sum_sq_val, count = 0.0, 0.0, 0

        for meta in self.demo_meta:
            arr = f[f"{meta['demo_key']}/depth"][...]
            flat = arr.flatten()
            count += flat.size
            min_val = min(min_val, flat.min())
            max_val = max(max_val, flat.max())
            sum_val += flat.sum()
            sum_sq_val += (flat ** 2).sum()

        if self.depth_normalization_method == 'minmax':
            range_val = max(max_val - min_val, 1)
            return {'method': 'minmax', 'min': min_val, 'range': range_val}
        else:
            mean = sum_val / count
            var = (sum_sq_val / count) - mean ** 2
            std = np.sqrt(var) if var > 0 else 1
            return {'method': self.depth_normalization_method, 'mean': mean, 'std': std}

    def __len__(self):
        if self.is_regression:
            return len(self.demo_meta)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Open file per worker lazily
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        if self.is_regression:
            meta = self.demo_meta[idx]
        elif self.is_waypointPlusTimings:

            demo_idx, t = self.valid_indices[idx]
            meta = self.demo_meta[demo_idx]

            target_timing = self.h5_file[f"{meta['demo_key']}/timings"][t + 1].astype(np.int8)-1  # Convert to zero-based index
            start_idx = max(0, t - self.sequence_length + 1)
            obs_seq = self._load_obs(meta['demo_key'], start_idx, t + 1)
            action_seq = self.h5_file[f"{meta['demo_key']}/path"][start_idx: t + 1].astype(np.float32)
            waypoints = np.array(self.h5_file[f"{meta['demo_key']}/ways"], dtype=np.float32)
            timings_sequence = self.h5_file[f"{meta['demo_key']}/timings"][start_idx: t + 1].astype(np.float32)

            if self.normalize_actions:
                action_seq = self._normalize_actions(action_seq)
                #target_action = self._normalize_actions(target_action)
            if self.observation_mode == 'depth' and self.normalize_depth:
                obs_seq = np.stack([self._normalize_depth(o) for o in obs_seq])

            # TODO normalize waypoint if needed

            # padding
            actual_len = len(obs_seq)
            pad_len = self.sequence_length - actual_len
            obs_pad_shape = (pad_len,) + obs_seq.shape[1:]
            obs_pad = np.zeros(obs_pad_shape, dtype=obs_seq.dtype)
            act_pad = np.zeros((pad_len, self.action_dim), dtype=action_seq.dtype)
            time_pad = np.zeros(pad_len, dtype=np.int64)

            obs_seq = np.concatenate([obs_pad, obs_seq], axis=0)
            action_seq = np.concatenate([act_pad, action_seq], axis=0)
            timing_seq = np.concatenate([time_pad, timings_sequence], axis=0)
            attention_mask = np.zeros(self.sequence_length, dtype=bool)
            attention_mask[-actual_len:] = True


            return {
                'observation': torch.from_numpy(obs_seq).float(),
                'previous_actions': torch.from_numpy(action_seq).float(),
                'action': torch.tensor(target_timing, dtype=torch.long),               
                'demo_id': torch.tensor(meta['demo_id'], dtype=torch.long),
                'waypoint': torch.from_numpy(waypoints).float(),
                #'previous_timings': torch.from_numpy(timing_seq).int()

            }

        else:
        # sequential mode
            demo_idx, t = self.valid_indices[idx]
            meta = self.demo_meta[demo_idx]

            target_action = self.h5_file[f"{meta['demo_key']}/path"][t + 1].astype(np.float32)
            start_idx = max(0, t - self.sequence_length + 1)
            obs_seq = self._load_obs(meta['demo_key'], start_idx, t + 1)
            action_seq = self.h5_file[f"{meta['demo_key']}/path"][start_idx: t + 1].astype(np.float32)

            if self.normalize_actions:
                action_seq = self._normalize_actions(action_seq)
                target_action = self._normalize_actions(target_action)
            if self.observation_mode == 'depth' and self.normalize_depth:
                obs_seq = np.stack([self._normalize_depth(o) for o in obs_seq])

            # padding
            actual_len = len(obs_seq)
            pad_len = self.sequence_length - actual_len
            obs_pad_shape = (pad_len,) + obs_seq.shape[1:]
            obs_pad = np.zeros(obs_pad_shape, dtype=obs_seq.dtype)
            act_pad = np.zeros((pad_len, self.action_dim), dtype=action_seq.dtype)
            time_pad = np.zeros(pad_len, dtype=np.int64)
            obs_seq = np.concatenate([obs_pad, obs_seq], axis=0)
            action_seq = np.concatenate([act_pad, action_seq], axis=0)
            timestep_seq = np.concatenate([time_pad, np.arange(start_idx, t + 1)], axis=0)
            attention_mask = np.zeros(self.sequence_length, dtype=bool)
            attention_mask[-actual_len:] = True

            return {
                'observation': torch.from_numpy(obs_seq).float(),
                'previous_actions': torch.from_numpy(action_seq).float(),
                'action': torch.from_numpy(target_action).float(),
                'timestep': torch.from_numpy(timestep_seq).long(),
                'attention_mask': torch.from_numpy(attention_mask),
                'demo_id': torch.tensor(meta['demo_id'], dtype=torch.long)
            }

    def _load_obs(self, demo_key, start=None, end=None):
        if self.observation_mode == 'depth':
            if start is None:
                return self.h5_file[f"{demo_key}/depth"][...].astype(np.float32)
            return self.h5_file[f"{demo_key}/depth"][start:end].astype(np.float32)
        elif self.observation_mode == 'points':
            if start is None:
                return self.h5_file[f"{demo_key}/points"][...].astype(np.float32)
            return self.h5_file[f"{demo_key}/points"][start:end].astype(np.float32)
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode}")

    def _normalize_actions(self, actions):
        if self.action_stats['method'] == 'minmax':
            return (actions - self.action_stats['min']) / self.action_stats['range']
        else:
            return (actions - self.action_stats['mean']) / self.action_stats['std']

    def _normalize_depth(self, depth):
        if self.depth_stats['method'] == 'minmax':
            return (depth - self.depth_stats['min']) / self.depth_stats['range']
        else:
            return (depth - self.depth_stats['mean']) / self.depth_stats['std']


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
    is_waypointPlusTimings: bool = False,
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
        is_waypointPlusTimings=is_waypointPlusTimings,
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
        is_waypointPlusTimings=is_waypointPlusTimings,
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
        is_waypointPlusTimings=data_cfg.get('is_waypointPlusTimings', False),
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