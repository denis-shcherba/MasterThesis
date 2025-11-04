import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

class ManipulationDataset(Dataset):
    """
    Streaming HDF5 dataset for imitation learning.
    """
    def __init__(
        self,
        h5_file_path: str,
        is_regression: bool = False,
        is_waypointPlusTimings: bool = False,
        sequence_length: int = 1,
        future_sequence_length: int = None,
        action_dim: int = 9,
        num_points: int = 1000,
        normalize_depth: bool = True,
        normalize_actions: bool = True,
        augment_data: bool = False,
        depth_dropout_prob: float = 0.05,
        depth_noise_scale: float = 0.0001,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',
        random_seed: int = 42,
        observation_mode: str = 'depth',
        depth_normalization_method: str = 'minmax',
        action_normalization_method: str = 'minmax',
        normalize_action_indices: Optional[List[int]] = None,  
    ):
        self.h5_file_path = h5_file_path
        self.is_regression = is_regression
        self.is_waypointPlusTimings = is_waypointPlusTimings
        self.sequence_length = sequence_length
        self.future_sequence_length = future_sequence_length if future_sequence_length is not None else sequence_length
        self.action_dim = action_dim
        self.num_points = num_points
        self.normalize_depth = normalize_depth
        self.normalize_actions = normalize_actions
        self.augment_data = augment_data
        self.split = split
        self.observation_mode = observation_mode
        self.depth_normalization_method = depth_normalization_method
        self.action_normalization_method = action_normalization_method
        self.normalize_action_indices = normalize_action_indices  # NEW

        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(random_seed)

        if self.augment_data and self.split == 'train':
            self.logger.info(f"Applying depth augmentation with dropout={depth_dropout_prob} and noise_scale={depth_noise_scale}")
        self.depth_dropout_prob = depth_dropout_prob
        self.depth_noise_scale = depth_noise_scale
        
        self.demo_meta: List[Dict] = []
        self.valid_indices: List[Tuple[int, int]] = []
        self._index_demonstrations()
        if self.split in ['train', 'val']:
            self._create_split(train_split)
        self._subsample_if_needed(subsample_demos)
        with h5py.File(self.h5_file_path, 'r') as f:
            if self.normalize_actions:
                self.action_stats = self._compute_action_normalization_stats(f)
            else:
                self.action_stats = None
            if self.normalize_depth and self.observation_mode == 'depth':
                self.depth_stats = self._compute_depth_normalization_stats(f)
            else:
                self.depth_stats = None

    def _augment_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """Applies domain randomization noise to a single depth image."""
        augmented_image = depth_image.copy()

        if self.depth_noise_scale > 0:
            valid_mask = augmented_image > 0
            std_dev = self.depth_noise_scale * (augmented_image[valid_mask] ** 2)
            noise = self.rng.normal(loc=0.0, scale=std_dev)
            augmented_image[valid_mask] += noise

        if self.depth_dropout_prob > 0:
            dropout_mask = self.rng.random(augmented_image.shape) < self.depth_dropout_prob
            augmented_image[dropout_mask] = 0.0

        return np.maximum(augmented_image, 0)

    def _augment_depth_image_realistic(
        self,
        depth_image: np.ndarray,
        dropout_patch_size: float = 5.0
    ) -> np.ndarray:
        augmented_image = depth_image.copy()

        if self.depth_noise_scale > 0:
            valid_mask = augmented_image > 0
            std_dev = self.depth_noise_scale * (augmented_image[valid_mask] ** 2)
            noise = self.rng.normal(loc=0.0, scale=std_dev)
            augmented_image[valid_mask] += noise

        if self.depth_dropout_prob > 0:
            random_noise = self.rng.random(augmented_image.shape)
            correlated_noise = gaussian_filter(random_noise, sigma=dropout_patch_size)
            dropout_mask = correlated_noise < np.percentile(correlated_noise, self.depth_dropout_prob * 100)
            augmented_image[dropout_mask] = 0.0

        return np.maximum(augmented_image, 0)

    def _index_demonstrations(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')], key=lambda x: int(x.split('_')[1]))
            for demo_idx, demo_key in enumerate(demo_keys):
                path_shape = f[f'{demo_key}/path'].shape
                if self.observation_mode == 'depth':
                    obs_shape = f[f'{demo_key}/depth'].shape
                else:
                    obs_shape = f[f'{demo_key}/points'].shape
                
                if len(path_shape) != 2 or path_shape[1] != self.action_dim or obs_shape[0] != path_shape[0]:
                    continue
                
                num_timesteps = path_shape[0]
                meta = {'demo_id': demo_idx, 'demo_key': demo_key, 'num_timesteps': num_timesteps}
                self.demo_meta.append(meta)
                
                if num_timesteps >= self.future_sequence_length:
                    for t in range(num_timesteps - self.future_sequence_length + 1):
                        self.valid_indices.append((demo_idx, t))
                
        if not self.demo_meta:
            raise ValueError(f"No valid demonstrations found in {self.h5_file_path}")

    def _create_split(self, train_split: float):
        """Filter demos for train/val split."""
        num_demos = len(self.demo_meta)
        indices = np.arange(num_demos)
        self.rng.shuffle(indices)
        split_idx = int(num_demos * train_split)
        
        if self.split == 'train':
            selected_demo_indices = set(indices[:split_idx])
        else:
            selected_demo_indices = set(indices[split_idx:])
        
        self.demo_meta = [meta for i, meta in enumerate(self.demo_meta) if i in selected_demo_indices]
        self.valid_indices = [(demo_idx, t) for demo_idx, t in self.valid_indices if demo_idx in selected_demo_indices]

        old_to_new_id_map = {old_meta['demo_id']: new_id for new_id, old_meta in enumerate(self.demo_meta)}
        
        for meta in self.demo_meta:
            meta['demo_id'] = old_to_new_id_map[meta['demo_id']]
            
        self.valid_indices = [(old_to_new_id_map[demo_idx], t) for demo_idx, t in self.valid_indices]

    def _subsample_if_needed(self, subsample_demos: Optional[int]):
        if subsample_demos is not None and subsample_demos > 0:
            if len(self.demo_meta) > subsample_demos:
                self.demo_meta = self.demo_meta[:subsample_demos]
                kept_demo_ids = {meta['demo_id'] for meta in self.demo_meta}
                self.valid_indices = [(demo_idx, t) for demo_idx, t in self.valid_indices if demo_idx in kept_demo_ids]

    def _compute_action_normalization_stats(self, f):
        # Determine which indices to compute stats for
        if self.normalize_action_indices is not None:
            indices_to_normalize = self.normalize_action_indices
        else:
            indices_to_normalize = list(range(self.action_dim))
        
        # Initialize arrays for computing stats only on normalized indices
        min_vals = np.full(self.action_dim, np.inf, dtype=np.float64)
        max_vals = np.full(self.action_dim, -np.inf, dtype=np.float64)
        sum_vals = np.zeros(self.action_dim, dtype=np.float64)
        sum_sq_vals = np.zeros(self.action_dim, dtype=np.float64)
        count = 0
        
        demo_ids_in_split = {meta['demo_id'] for meta in self.demo_meta}
        original_demo_meta = []
        with h5py.File(self.h5_file_path, 'r') as temp_f:
            all_demo_keys = sorted([k for k in temp_f.keys() if k.startswith('demo_')], key=lambda x: int(x.split('_')[1]))
            original_demo_meta = [{'demo_key': key} for idx, key in enumerate(all_demo_keys) if idx in demo_ids_in_split]
        
        for meta in original_demo_meta:
            arr = f[f"{meta['demo_key']}/path"][...]
            count += arr.shape[0]
            
            # Only compute stats for specified indices
            for idx in indices_to_normalize:
                min_vals[idx] = min(min_vals[idx], arr[:, idx].min())
                max_vals[idx] = max(max_vals[idx], arr[:, idx].max())
                sum_vals[idx] += arr[:, idx].sum()
                sum_sq_vals[idx] += (arr[:, idx] ** 2).sum()

        if self.action_normalization_method == 'minmax':
            range_vals = np.ones(self.action_dim, dtype=np.float64)
            # Compute range only for normalized indices
            for idx in indices_to_normalize:
                range_vals[idx] = max_vals[idx] - min_vals[idx] if max_vals[idx] - min_vals[idx] != 0 else 1.0
            
            # For unnormalized indices, set neutral values: min=0, range=1
            for idx in range(self.action_dim):
                if idx not in indices_to_normalize:
                    min_vals[idx] = 0.0
                    range_vals[idx] = 1.0
            
            return {
                'method': 'minmax',
                'min': min_vals.tolist(),
                'range': range_vals.tolist(),
                'normalize_indices': list(indices_to_normalize)
            }
        else:
            mean = np.zeros(self.action_dim, dtype=np.float64)
            std = np.ones(self.action_dim, dtype=np.float64)
            
            # Compute mean and std only for normalized indices
            for idx in indices_to_normalize:
                mean[idx] = sum_vals[idx] / count
                var = (sum_sq_vals[idx] / count) - mean[idx] ** 2
                std[idx] = np.sqrt(var) if var > 1e-8 else 1.0
            
            # Unnormalized indices already have mean=0, std=1
            
            return {
                'method': self.action_normalization_method,
                'mean': mean.tolist(),
                'std': std.tolist(),
                'normalize_indices': list(indices_to_normalize)
            }

    def _compute_depth_normalization_stats(self, f):
        min_val, max_val = np.inf, -np.inf
        sum_val, sum_sq_val, count = 0.0, 0.0, 0

        demo_ids_in_split = {meta['demo_id'] for meta in self.demo_meta}
        original_demo_meta = []
        with h5py.File(self.h5_file_path, 'r') as temp_f:
            all_demo_keys = sorted([k for k in temp_f.keys() if k.startswith('demo_')], key=lambda x: int(x.split('_')[1]))
            original_demo_meta = [{'demo_key': key} for idx, key in enumerate(all_demo_keys) if idx in demo_ids_in_split]
        for meta in original_demo_meta:
            arr = f[f"{meta['demo_key']}/depth"][...]
            flat = arr.flatten()
            count += flat.size
            min_val = min(min_val, flat.min())
            max_val = max(max_val, flat.max())
            sum_val += flat.sum()
            sum_sq_val += (flat ** 2).sum()

        if self.depth_normalization_method == 'minmax':
            range_val = max(max_val - min_val, 1e-8)
            return {'method': 'minmax', 'min': float(min_val), 'range': float(range_val)}
        else:
            mean = sum_val / count
            var = (sum_sq_val / count) - mean ** 2
            std = np.sqrt(var) if var > 1e-8 else 1
            return {'method': self.depth_normalization_method, 'mean': float(mean), 'std': float(std)}
            
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        MODIFIED: Fetches data and applies left-padding to history sequences
        if they fall before the start of a demonstration.
        Ensures at least the current observation is always included.
        """
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True)

        demo_idx, future_start_t = self.valid_indices[idx]
        meta = self.demo_meta[demo_idx]
        
        # 1. --- Handle the FUTURE (target) sequence ---
        future_end_t = future_start_t + self.future_sequence_length
        future_actions_sequence = self.h5_file[f"{meta['demo_key']}/path"][future_start_t:future_end_t].astype(np.float32)

        # 2. --- Handle the PAST (history) sequence with PADDING ---
        # Include current observation (future_start_t) in history
        history_end_t = future_start_t + 1
        history_start_t = history_end_t - self.sequence_length
        
        num_to_pad = max(0, -history_start_t)
        num_to_fetch = self.sequence_length - num_to_pad
        
        obs_sample_shape = self._load_obs(meta['demo_key'], 0, 1).shape[1:]
        obs_sequence = np.zeros((self.sequence_length, *obs_sample_shape), dtype=np.float32)
        past_actions_sequence = np.zeros((self.sequence_length, self.action_dim), dtype=np.float32)
        
        # num_to_fetch is always >= 1 (at minimum the current observation)
        real_data_start_t = max(0, history_start_t)
        
        real_obs = self._load_obs(meta['demo_key'], real_data_start_t, history_end_t)
        real_actions = self.h5_file[f"{meta['demo_key']}/path"][real_data_start_t:history_end_t].astype(np.float32)
        
        if self.augment_data and self.split == 'train' and self.observation_mode == 'depth':
            augmented_obs = np.array([self._augment_depth_image(img) for img in real_obs])
            real_obs = augmented_obs

        obs_sequence[num_to_pad:] = real_obs
        past_actions_sequence[num_to_pad:] = real_actions

        # 3. --- Handle Normalization ---
        if self.normalize_actions:
            past_actions_sequence[num_to_pad:] = self._normalize_actions(past_actions_sequence[num_to_pad:])
            future_actions_sequence = self._normalize_actions(future_actions_sequence)
        
        if self.observation_mode == 'depth' and self.normalize_depth:
            normalized_obs = np.array([self._normalize_depth(img) for img in obs_sequence[num_to_pad:]])
            obs_sequence[num_to_pad:] = normalized_obs

        # 4. --- Load metadata and convert to tensors ---
        book_params = self.h5_file[f"{meta['demo_key']}/book_params"][...].astype(np.float32) if 'book_params' in self.h5_file[f"{meta['demo_key']}"] else np.array([0.0], dtype=np.float32)
        
        return {
            'observation_sequence': torch.from_numpy(obs_sequence).float(),
            'previous_actions_sequence': torch.from_numpy(past_actions_sequence).float(),
            'target_actions_sequence': torch.from_numpy(future_actions_sequence).float(),
            'demo_id': torch.tensor(meta['demo_id'], dtype=torch.long),
            'book_params': torch.from_numpy(book_params).float(),
        }
    
    def _load_obs(self, demo_key, start=None, end=None):
        key = 'depth' if self.observation_mode == 'depth' else 'points'
        if start is None:
            return self.h5_file[f"{demo_key}/{key}"][...].astype(np.float32)
        return self.h5_file[f"{demo_key}/{key}"][start:end].astype(np.float32)

    def _normalize_actions(self, actions):
        stats = self.action_stats
        normalized_actions = actions.copy()
        
        # Get indices to normalize
        indices_to_normalize = stats.get('normalize_indices', list(range(self.action_dim)))
        
        if stats['method'] == 'minmax':
            for idx in indices_to_normalize:
                normalized_actions[..., idx] = 2 * (actions[..., idx] - stats['min'][idx]) / stats['range'][idx] - 1
        else:  # zscore
            for idx in indices_to_normalize:
                normalized_actions[..., idx] = (actions[..., idx] - stats['mean'][idx]) / stats['std'][idx]
        
        return normalized_actions

    def _normalize_depth(self, depth):
        stats = self.depth_stats
        if stats['method'] == 'minmax':
            return (depth - stats['min']) / stats['range']
        else:
            return (depth - stats['mean']) / stats['std']

def create_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    future_sequence_length: int = None,
    action_dim: int = 9,
    num_points: int = 1000,
    train_split: float = 0.8,
    augment_data: bool = True,
    depth_dropout_prob: float = 0.05,
    depth_noise_scale: float = 0.0001,
    num_workers: int = 4,
    subsample_demos: Optional[int] = None,
    random_seed: int = 42,
    is_regression: bool = False,
    is_waypointPlusTimings: bool = False,
    observation_mode: str = 'points',
    normalize_depth: bool = True,
    normalize_actions: bool = True,
    depth_normalization_method = 'minmax',
    action_normalization_method = 'zscore',
    normalize_action_indices: Optional[List[int]] = None

) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        future_sequence_length=future_sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        augment_data=augment_data,
        depth_dropout_prob=depth_dropout_prob,
        depth_noise_scale=depth_noise_scale,
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
        action_normalization_method=action_normalization_method,
        normalize_action_indices=normalize_action_indices
    )

    val_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        future_sequence_length=future_sequence_length,
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
        action_normalization_method=action_normalization_method,
        normalize_action_indices=normalize_action_indices
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
        future_sequence_length=data_cfg.get('future_sequence_length', None),
        action_dim=data_cfg.action_dim,
        num_points=data_cfg.get('num_points', 0),
        train_split=data_cfg.train_split,
        normalize_depth=data_cfg.get('normalize_depth', True),
        normalize_actions=data_cfg.get('normalize_actions', True),
        augment_data=data_cfg.get('augment_data', False),
        depth_dropout_prob=data_cfg.get('depth_dropout_prob', 0.05),
        depth_noise_scale=data_cfg.get('depth_noise_scale', 0.0001),
        num_workers=data_cfg.num_workers,
        subsample_demos=data_cfg.get('subsample_demos', None),
        random_seed=data_cfg.random_seed,
        is_regression=data_cfg.get('is_regression', False),
        is_waypointPlusTimings=data_cfg.get('is_waypointPlusTimings', False),
        observation_mode=cfg.get('observation_mode', 'points'), 
        depth_normalization_method=data_cfg.get('depth_normalization_method', 'minmax'),
        action_normalization_method=data_cfg.get('action_normalization_method', 'zscore'),
        normalize_action_indices=data_cfg.get('normalize_action_indices', None),
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