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
        depth_dropout_prob: float = 0.05,
        depth_noise_scale: float = 0.0001,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',
        random_seed: int = 42,
        observation_mode: str = 'depth',
        depth_normalization_method: str = 'minmax',
        action_normalization_method: str = 'minmax',
        normalize_action_indices: Optional[List[int]] = None
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
        self.split = split
        self.observation_mode = observation_mode
        self.depth_normalization_method = depth_normalization_method
        self.action_normalization_method = action_normalization_method
        self.normalize_action_indices = normalize_action_indices

        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(random_seed)

        if self.observation_mode == 'depth':
            self.obs_key = 'depth'

        elif self.observation_mode == 'dino_cls':
            self.obs_key = 'cls_features'
            self.normalize_depth = False
            print("INFO: Using pre-extracted DINO CLS features. "
                  "Depth normalization disabled.")

        elif self.observation_mode == 'dino_patches':
            # (T, num_patches, dim) or (T, H_p, W_p, dim)
            self.obs_key = 'patch_features'
            self.normalize_depth = False
            print("INFO: Using pre-extracted DINO PATCH features "
                  "(no depth normalization).")

        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

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

        if self.observation_mode in ['dino_cls', 'dino_patches', 'depth']:
            print(f"Starting preload for {self.split} split...")
            self._preload_all_data()  # Preload everything
            print(f"Preload complete for {self.split} split")
            
    def _preload_all_data(self):
        """Preload ALL data into RAM - called once during init"""
        self.feature_cache = {}
        self.action_cache = {}
        self.book_params_cache = {}
        self.waypoint_cache = {}
        self.initial_obs_cache = {}
        
        total_size = 0
        
        with h5py.File(self.h5_file_path, 'r') as f:
            for meta in self.demo_meta:
                demo_key = meta['demo_key']
                
                # Preload features
                self.feature_cache[demo_key] = f[f"{demo_key}/{self.obs_key}"][...].astype(np.float32)
                total_size += self.feature_cache[demo_key].nbytes
                
                # Preload actions
                self.action_cache[demo_key] = f[f"{demo_key}/path"][...].astype(np.float32)
                total_size += self.action_cache[demo_key].nbytes
                
                # Preload metadata
                if 'book_params' in f[demo_key]:
                    self.book_params_cache[demo_key] = f[f"{demo_key}/book_params"][...].astype(np.float32)
                else:
                    self.book_params_cache[demo_key] = np.array([0.0], dtype=np.float32)
                
                if 'waypoints' in f[demo_key]:
                    self.waypoint_cache[demo_key] = f[f"{demo_key}/waypoints"][...].astype(np.float32)
                else:
                    self.waypoint_cache[demo_key] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                self.initial_obs_cache[demo_key] = self.feature_cache[demo_key][0]
        
        print(f"✓ Preloaded {len(self.feature_cache)} demos: {total_size/1e9:.2f} GB in RAM")
    
        # Don't need HDF5 file anymore - prevent any disk access
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def _index_demonstrations(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')], key=lambda x: int(x.split('_')[1]))
            
            for demo_idx, demo_key in enumerate(demo_keys):
                path_shape = f[f'{demo_key}/path'].shape
                
                obs_data_key = f'{demo_key}/{self.obs_key}'
                if obs_data_key not in f:
                    self.logger.warning(f"Skipping demo {demo_key}: Observation key '{self.obs_key}' not found.")
                    continue
                
                obs_shape = f[obs_data_key].shape
                
                # Check 1: path data is 2D and action_dim matches
                # Check 2: observation (time) steps match action (time) steps
                if len(path_shape) != 2 or path_shape[1] != self.action_dim or obs_shape[0] != path_shape[0]:
                    self.logger.warning(f"Skipping demo {demo_key}: Shape mismatch. Path: {path_shape}, Obs: {obs_shape}")
                    continue
                
                # Save the full observation shape (e.g., (768,) for CLS features)
                obs_sample_shape = obs_shape[1:] 
                
                num_timesteps = path_shape[0]
                meta = {'demo_id': demo_idx, 'demo_key': demo_key, 'num_timesteps': num_timesteps, 'obs_sample_shape': obs_sample_shape} # <-- Store shape
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
        """Now fully in-memory - no HDF5 access"""
        demo_idx, future_start_t = self.valid_indices[idx]
        meta = self.demo_meta[demo_idx]
        demo_key = meta['demo_key']
        
        # === ALL FROM CACHE - NO DISK I/O ===
        future_end_t = future_start_t + self.future_sequence_length
        future_actions_sequence = self.action_cache[demo_key][future_start_t:future_end_t].copy()

        history_end_t = future_start_t + 1
        history_start_t = history_end_t - self.sequence_length
        num_to_pad = max(0, -history_start_t)
        
        obs_sample_shape = meta['obs_sample_shape']
        obs_sequence = np.zeros((self.sequence_length, *obs_sample_shape), dtype=np.float32)
        past_actions_sequence = np.zeros((self.sequence_length, self.action_dim), dtype=np.float32)
        
        real_data_start_t = max(0, history_start_t)
        
        # Load from CACHE (not HDF5!)
        real_obs = self.feature_cache[demo_key][real_data_start_t:history_end_t]
        real_actions = self.action_cache[demo_key][real_data_start_t:history_end_t]
        
        obs_sequence[num_to_pad:] = real_obs
        past_actions_sequence[num_to_pad:] = real_actions

        # Normalization...
        if self.normalize_actions:
            past_actions_sequence[num_to_pad:] = self._normalize_actions(past_actions_sequence[num_to_pad:])
            future_actions_sequence = self._normalize_actions(future_actions_sequence)
        
        if self.observation_mode == 'depth' and self.normalize_depth:
            normalized_obs = np.array([self._normalize_depth(img) for img in obs_sequence[num_to_pad:]])
            obs_sequence[num_to_pad:] = normalized_obs

        # Load metadata from CACHE (not HDF5!)
        book_params = self.book_params_cache[demo_key]
        waypoint = self.waypoint_cache[demo_key]
        initial_obs = self.initial_obs_cache[demo_key]

        return {
            'observation_sequence': torch.from_numpy(obs_sequence).float(),
            'previous_actions_sequence': torch.from_numpy(past_actions_sequence).float(),
            'target_actions_sequence': torch.from_numpy(future_actions_sequence).float(),
            'demo_id': torch.tensor(meta['demo_id'], dtype=torch.long),
            'book_params': torch.from_numpy(book_params).float(),
            'waypoint': torch.from_numpy(waypoint).float(),
            'initial_observation': torch.from_numpy(initial_obs).float()
        }
    
    def _load_obs(self, demo_key, start=None, end=None):
        if hasattr(self, 'feature_cache') and demo_key in self.feature_cache:
            if start is None:
                return self.feature_cache[demo_key]
            return self.feature_cache[demo_key][start:end]
        
        # Fallback to HDF5
        obs_key_path = f"{demo_key}/{self.obs_key}"
        if start is None:
            return self.h5_file[obs_key_path][...].astype(np.float32)
        return self.h5_file[obs_key_path][start:end].astype(np.float32)

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
        drop_last=True,
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
