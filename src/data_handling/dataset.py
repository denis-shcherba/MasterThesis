import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

from data_handling.waypoint_dataset import create_wp_dataloaders

class ManipulationDataset(Dataset):
    """
    Streaming HDF5 dataset for imitation learning with Augmentation support.
    Supports optional observation caching for memory management.
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
        normalize_obs: bool = True,
        normalize_actions: bool = True,
        subsample_demos: Optional[int] = None,
        train_split: float = 0.8,
        split: str = 'train',
        random_seed: int = 42,
        observation_mode: str = 'depth',
        depth_normalization_method: str = 'minmax',
        action_normalization_method: str = 'minmax',
        normalize_action_indices: Optional[List[int]] = None,
        cache_observations: bool = False,
        # NEW ARGUMENT: Allows passing stats from main process to workers
        precomputed_stats: Optional[Dict] = None, 
    ):
        self.h5_file_path = h5_file_path
        self.is_regression = is_regression
        self.is_waypointPlusTimings = is_waypointPlusTimings
        self.sequence_length = sequence_length
        self.future_sequence_length = future_sequence_length if future_sequence_length is not None else sequence_length
        self.action_dim = action_dim
        self.num_points = num_points
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.split = split
        self.observation_mode = observation_mode
        self.depth_normalization_method = depth_normalization_method
        self.action_normalization_method = action_normalization_method
        self.normalize_action_indices = normalize_action_indices
        self.cache_observations = cache_observations

        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(random_seed)

        # --- Observation Mode Setup ---
        if self.observation_mode == 'depth':
            self.obs_key = 'depth'
        elif self.observation_mode == 'rgb':
            self.obs_key = 'rgb' 
        elif self.observation_mode == 'points':
            self.normalize_obs = False
            self.obs_key = 'points'
        elif self.observation_mode == 'dino_cls':
            self.obs_key = 'cls_features'
            self.normalize_obs = False
        elif self.observation_mode == 'dino_patches':
            self.obs_key = 'patch_features'
            self.normalize_obs = False
        elif self.observation_mode == 'sam_points':
            self.obs_key = 'points'
            self.normalize_obs = False
        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

        
        self.demo_meta: List[Dict] = []
        self.valid_indices: List[Tuple[int, int]] = []
        
        # 1. Index all available keys and parse augmentation info
        self._index_demonstrations()
        
        # 2. Create Train/Val split (Handles augmentation logic)
        if self.split in ['train', 'val']:
            self._create_split(train_split)
            
        # 3. Subsample if requested (on top of the split)
        self._subsample_if_needed(subsample_demos)

        # 4. Compute (or Load) Stats
        self.action_stats = None
        self.obs_stats = None

        if precomputed_stats is not None:
            # WORKER PATH: Use the stats provided by the main process (Fast!)
            self.action_stats = precomputed_stats.get('action_stats')
            self.obs_stats = precomputed_stats.get('obs_stats')
        else:
            # MAIN PROCESS PATH: Compute stats from disk (Slow, done once)
            # Only open file if we actually need to compute something
            if self.normalize_actions or (self.normalize_obs and self.observation_mode == 'depth'):
                with h5py.File(self.h5_file_path, 'r') as f:
                    if self.normalize_actions:
                        self.action_stats = self._compute_action_normalization_stats(f)
                    
                    if self.normalize_obs and self.observation_mode == 'depth':
                        self.obs_stats = self._compute_depth_normalization_stats(f)

        # 5. Preload data into RAM
        self.h5_file = None  # Will be opened lazily per worker
        if self.observation_mode in ['dino_cls', 'dino_patches', 'depth', 'sam_points', 'points', 'rgb']:
            if self.cache_observations:
                print(f"[{self.split}] Starting preload (observations in RAM)...")
                self._preload_all_data()
                print(f"[{self.split}] Preload complete.")
            else:
                print(f"[{self.split}] Starting preload (observations from disk)...")
                self._preload_actions_only()
                print(f"[{self.split}] Preload complete.")
    def _index_demonstrations(self):
        """
        Scans HDF5, checks shapes, and populates self.demo_meta.
        Now also parses 'demo_0_aug_1' to identify base_id and is_augmented.
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            # Sort ensures demo_0 comes before demo_0_aug_1
            all_keys = sorted([k for k in f.keys() if k.startswith('demo_')])
            
            for demo_idx, demo_key in enumerate(all_keys):
                # --- Parse Augmentation Info ---
                parts = demo_key.split('_') # e.g. ['demo', '0'] or ['demo', '0', 'aug', '1']
                try:
                    base_id = int(parts[1])
                except ValueError:
                    self.logger.warning(f"Skipping {demo_key}: Could not parse ID.")
                    continue
                
                is_augmented = len(parts) > 2 and 'aug' in parts

                path_shape = f[f'{demo_key}/path'].shape
                obs_data_key = f'{demo_key}/{self.obs_key}'
                
                if obs_data_key not in f:
                    continue
                
                obs_shape = f[obs_data_key].shape
                
                if len(path_shape) != 2 or (obs_shape[0] != path_shape[0] and self.obs_key != 'points'):
                    continue

                obs_sample_shape = obs_shape[1:]
                num_timesteps = path_shape[0]
                
                meta = {
                    'demo_id': demo_idx,
                    'demo_key': demo_key,
                    'base_id': base_id,
                    'is_augmented': is_augmented,
                    'num_timesteps': num_timesteps, 
                    'obs_sample_shape': obs_sample_shape
                }
                self.demo_meta.append(meta)
                
                # Pre-calculate valid start indices for sliding windows
                if num_timesteps >= self.future_sequence_length:
                    for t in range(num_timesteps - self.future_sequence_length + 1):
                        self.valid_indices.append((demo_idx, t))

        if not self.demo_meta:
            raise ValueError(f"No valid demonstrations found in {self.h5_file_path}")

    def _create_split(self, train_split: float):
        """
        Splits data by BASE_ID.
        Train: Includes Base + Augmentations.
        Val: Includes Base ONLY (discards augmentations).
        """
        # 1. Identify all unique base IDs
        all_base_ids = sorted(list(set(m['base_id'] for m in self.demo_meta)))
        num_base = len(all_base_ids)
        
        # 2. Shuffle base IDs to create random split
        indices = np.arange(num_base)
        self.rng.shuffle(indices)
        
        split_idx = int(num_base * train_split)
        
        train_base_ids = set(all_base_ids[i] for i in indices[:split_idx])
        val_base_ids = set(all_base_ids[i] for i in indices[split_idx:])
        
        target_ids = train_base_ids if self.split == 'train' else val_base_ids
        
        # 3. Filter demo_meta
        new_demo_meta = []
        old_indices_kept = set()
        
        for meta in self.demo_meta:
            bid = meta['base_id']
            is_aug = meta['is_augmented']
            
            if bid in target_ids:
                if self.split == 'train':
                    new_demo_meta.append(meta)
                    old_indices_kept.add(meta['demo_id'])
                else:
                    if not is_aug:
                        new_demo_meta.append(meta)
                        old_indices_kept.add(meta['demo_id'])
        
        self.demo_meta = new_demo_meta
        
        # 4. Remap valid_indices
        old_to_new_id_map = {old_meta['demo_id']: new_id for new_id, old_meta in enumerate(self.demo_meta)}
        
        for meta in self.demo_meta:
            meta['demo_id'] = old_to_new_id_map[meta['demo_id']]
            
        self.valid_indices = [
            (old_to_new_id_map[old_idx], t) 
            for old_idx, t in self.valid_indices 
            if old_idx in old_indices_kept
        ]
        
        print(f"[{self.split.upper()}] Loaded {len(self.demo_meta)} trajectories "
              f"from {len(target_ids)} unique expert demos.")
    
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
                
                if self.observation_mode != 'sam_points':
                    self.initial_obs_cache[demo_key] = self.feature_cache[demo_key][0]
                else:
                    self.initial_obs_cache[demo_key] = self.feature_cache[demo_key]

        print(f"✓ Preloaded {len(self.feature_cache)} demos: {total_size/1e9:.2f} GB in RAM")
    
        # Don't need HDF5 file anymore - prevent any disk access
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def _preload_actions_only(self):
        """Preload actions and metadata only - observations loaded on-demand from disk"""
        self.feature_cache = None  # Signal that observations are not cached
        self.action_cache = {}
        self.book_params_cache = {}
        self.waypoint_cache = {}
        self.initial_obs_cache = {}
        
        total_size = 0
        
        with h5py.File(self.h5_file_path, 'r') as f:
            for meta in self.demo_meta:
                demo_key = meta['demo_key']
                
                # Preload actions (always small)
                self.action_cache[demo_key] = f[f"{demo_key}/path"][...].astype(np.float32)
                total_size += self.action_cache[demo_key].nbytes
                
                # Preload metadata (always small)
                if 'book_params' in f[demo_key]:
                    self.book_params_cache[demo_key] = f[f"{demo_key}/book_params"][...].astype(np.float32)
                else:
                    self.book_params_cache[demo_key] = np.array([0.0], dtype=np.float32)
                
                if 'waypoints' in f[demo_key]:
                    self.waypoint_cache[demo_key] = f[f"{demo_key}/waypoints"][...].astype(np.float32)
                else:
                    self.waypoint_cache[demo_key] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                # Preload initial observation (single frame, small)
                if self.observation_mode != 'sam_points':
                    self.initial_obs_cache[demo_key] = f[f"{demo_key}/{self.obs_key}"][0].astype(np.float32)
                else:
                    self.initial_obs_cache[demo_key] = f[f"{demo_key}/{self.obs_key}"][...].astype(np.float32)

        print(f"✓ Preloaded actions/metadata: {total_size/1e9:.2f} GB in RAM (observations on disk)")

    def _subsample_if_needed(self, subsample_demos: Optional[int]):
        if subsample_demos is not None and subsample_demos > 0:
            if len(self.demo_meta) > subsample_demos:
                self.demo_meta = self.demo_meta[:subsample_demos]
                kept_demo_ids = {meta['demo_id'] for meta in self.demo_meta}
                self.valid_indices = [(demo_idx, t) for demo_idx, t in self.valid_indices if demo_idx in kept_demo_ids]

    def _compute_action_normalization_stats(self, f):
        if self.normalize_action_indices is not None:
            indices_to_normalize = self.normalize_action_indices
        else:
            indices_to_normalize = list(range(self.action_dim))
        
        min_vals = np.full(self.action_dim, np.inf, dtype=np.float64)
        max_vals = np.full(self.action_dim, -np.inf, dtype=np.float64)
        sum_vals = np.zeros(self.action_dim, dtype=np.float64)
        sum_sq_vals = np.zeros(self.action_dim, dtype=np.float64)
        count = 0
        
        for meta in self.demo_meta:
            arr = f[f"{meta['demo_key']}/path"][...]
            count += arr.shape[0]
            
            for idx in indices_to_normalize:
                min_vals[idx] = min(min_vals[idx], arr[:, idx].min())
                max_vals[idx] = max(max_vals[idx], arr[:, idx].max())
                sum_vals[idx] += arr[:, idx].sum()
                sum_sq_vals[idx] += (arr[:, idx] ** 2).sum()

        if self.action_normalization_method == 'minmax':
            range_vals = np.ones(self.action_dim, dtype=np.float64)
            for idx in indices_to_normalize:
                range_vals[idx] = max_vals[idx] - min_vals[idx] if max_vals[idx] - min_vals[idx] != 0 else 1.0
            
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
            
            for idx in indices_to_normalize:
                mean[idx] = sum_vals[idx] / count
                var = (sum_sq_vals[idx] / count) - mean[idx] ** 2
                std[idx] = np.sqrt(var) if var > 1e-8 else 1.0
            
            return {
                'method': self.action_normalization_method,
                'mean': mean.tolist(),
                'std': std.tolist(),
                'normalize_indices': list(indices_to_normalize)
            }

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
            range_val = max(max_val - min_val, 1e-8)
            return {'method': 'minmax', 'min': float(min_val), 'range': float(range_val)}
        else:
            mean = sum_val / count
            var = (sum_sq_val / count) - mean ** 2
            std = np.sqrt(var) if var > 1e-8 else 1
            return {'method': self.depth_normalization_method, 'mean': float(mean), 'std': float(std)}

    def __len__(self):
        return len(self.valid_indices)

    def _get_h5_file(self):
        """Lazily open HDF5 file per worker process"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
        return self.h5_file

    def __getitem__(self, idx):
        demo_idx, future_start_t = self.valid_indices[idx]
        meta = self.demo_meta[demo_idx]
        demo_key = meta['demo_key']
        
        future_end_t = future_start_t + self.future_sequence_length
        
        # Actions always from cache
        future_actions_sequence = self.action_cache[demo_key][future_start_t:future_end_t][:, :self.action_dim].copy()

        history_end_t = future_start_t + 1
        history_start_t = history_end_t - self.sequence_length
        num_to_pad = max(0, -history_start_t)
        
        obs_sample_shape = meta['obs_sample_shape']
        
        # ---------------------------------------------------------
        # OPTIMIZATION: USE UINT8 BUFFER ONLY FOR RGB
        # ---------------------------------------------------------
        if self.observation_mode == 'rgb':
            dtype_to_use = np.uint8
        else:
            # Depth, Points, Features need float precision immediately
            dtype_to_use = np.float32

        # Initialize buffer with the chosen dtype
        obs_sequence = np.zeros((self.sequence_length, *obs_sample_shape), dtype=dtype_to_use)
        past_actions_sequence = np.zeros((self.sequence_length, self.action_dim), dtype=np.float32)
        
        real_data_start_t = max(0, history_start_t)
        
        # Get observations - from cache or disk
        if self.cache_observations:
            # Slicing a cache works for both uint8 and float32
            obs_sequence[num_to_pad:] = self.feature_cache[demo_key][real_data_start_t:history_end_t]
        else:
            h5_file = self._get_h5_file()
            # HDF5 read: If dataset is uint8 and buffer is uint8, this is fast and low memory.
            # If dataset is uint8 and buffer is float32 (Depth), this does an implicit cast (fine).
            obs_sequence[num_to_pad:] = h5_file[f"{demo_key}/{self.obs_key}"][real_data_start_t:history_end_t]
        
        # Actions always from cache
        real_actions = self.action_cache[demo_key][real_data_start_t:history_end_t][:, :self.action_dim]
        
        past_actions_sequence[num_to_pad:] = real_actions

        # Normalize Actions
        if self.normalize_actions:
            past_actions_sequence[num_to_pad:] = self._normalize_actions(past_actions_sequence[num_to_pad:])
            future_actions_sequence = self._normalize_actions(future_actions_sequence)
        
        # Fetch other metadata
        book_params = self.book_params_cache[demo_key]
        waypoint = self.waypoint_cache[demo_key]
        initial_obs = self.initial_obs_cache[demo_key] # This might be uint8 or float32 depending on mode

        # --- VISUAL NORMALIZATION & PERMUTATION LOGIC ---
        
        # Define a variable for the final float output
        final_obs_sequence = obs_sequence # Default alias

        if self.observation_mode in ['depth', 'rgb'] and self.normalize_obs:
            
            # 1. Normalize and Cast to Float
            # The list comprehension naturally outputs float arrays if _normalize_image returns floats
            normalized_seq = np.array([self._normalize_image(img) for img in obs_sequence[num_to_pad:]], dtype=np.float32)
            
            # Since we might have changed dtype (uint8 -> float32), we can't shove it back into obs_sequence
            # if obs_sequence was initialized as uint8. We need a new buffer or overwrite if compatible.
            
            if self.observation_mode == 'rgb':
                # Create a new float buffer for the final output
                final_obs_sequence = np.zeros((self.sequence_length, *obs_sample_shape), dtype=np.float32)
                final_obs_sequence[num_to_pad:] = normalized_seq
            else:
                # Depth was already float32, so we can overwrite in place
                obs_sequence[num_to_pad:] = normalized_seq
                final_obs_sequence = obs_sequence

            # 2. Normalize Initial Observation
            initial_obs = self._normalize_image(initial_obs).astype(np.float32)

            # 3. Handle RGB Transpose (HWC -> CHW)
            if self.observation_mode == 'rgb':
                # Permute ENTIRE sequence at once: (T, H, W, C) -> (T, C, H, W)
                final_obs_sequence = np.transpose(final_obs_sequence, (0, 3, 1, 2))
                
                # Permute initial observation: (H, W, C) -> (C, H, W)
                initial_obs = np.transpose(initial_obs, (2, 0, 1))

        return {
            'observation_sequence': torch.from_numpy(final_obs_sequence).float(),
            'previous_actions_sequence': torch.from_numpy(past_actions_sequence).float(),
            'target_actions_sequence': torch.from_numpy(future_actions_sequence).float(),
            'demo_id': torch.tensor(meta['demo_id'], dtype=torch.long),
            'book_params': torch.from_numpy(book_params).float(),
            'waypoint': torch.from_numpy(waypoint).float(),
            'initial_observation': torch.from_numpy(initial_obs).float()
        }
    
    def _normalize_actions(self, actions):
        stats = self.action_stats
        normalized_actions = actions.copy()
        indices_to_normalize = stats.get('normalize_indices', list(range(self.action_dim)))
        
        if stats['method'] == 'minmax':
            for idx in indices_to_normalize:
                normalized_actions[..., idx] = 2 * (actions[..., idx] - stats['min'][idx]) / stats['range'][idx] - 1
        else:  # zscore
            for idx in indices_to_normalize:
                normalized_actions[..., idx] = (actions[..., idx] - stats['mean'][idx]) / stats['std'][idx]
        return normalized_actions

    def _normalize_image(self, img):
            """
            Handles generic image normalization.
            img: np.ndarray (H, W) for depth or (H, W, C) for RGB
            """
            if self.observation_mode == 'rgb':
                # CASE 1: RGB
                # Standard: Scale 0-255 to 0-1
                img = img / 255.0
                
                # Optional: If using Pretrained ResNet, typically apply ImageNet Mean/Std here
                # mean = np.array([0.485, 0.456, 0.406])
                # std = np.array([0.229, 0.224, 0.225])
                # img = (img - mean) / std
                
                return img

            elif self.observation_mode == 'depth':
                # CASE 2: Depth (Uses pre-computed stats)
                stats = self.obs_stats
                if stats['method'] == 'minmax':
                    # Clip to ensure we don't go out of bounds on validation data
                    val = np.clip(img, stats['min'], stats['min'] + stats['range'])
                    return (val - stats['min']) / stats['range']
                else:
                    return (img - stats['mean']) / stats['std']
            
            return img
    
    def __del__(self):
        """Clean up HDF5 file handle if it's open"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass

def create_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    future_sequence_length: int = None,
    action_dim: int = 9,
    num_points: int = 1000,
    train_split: float = 0.8,
    num_workers: int = 4,
    subsample_demos: Optional[int] = None,
    random_seed: int = 42,
    is_regression: bool = False,
    is_waypointPlusTimings: bool = False,
    observation_mode: str = 'points',
    normalize_obs: bool = True,
    normalize_actions: bool = True,
    depth_normalization_method = 'minmax',
    action_normalization_method = 'zscore',
    normalize_action_indices: Optional[List[int]] = None
) -> Tuple[DataLoader, DataLoader]:
    
    # 1. Train Dataset (Include Augmentations)
    print("Initializing Train Dataset (Computing Stats)...")
    train_dataset = ManipulationDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        future_sequence_length=future_sequence_length,
        action_dim=action_dim,
        num_points=num_points,
        subsample_demos=subsample_demos,
        train_split=train_split,
        split='train',
        random_seed=random_seed,
        is_regression=is_regression,
        is_waypointPlusTimings=is_waypointPlusTimings,
        observation_mode=observation_mode,
        normalize_obs=normalize_obs,
        normalize_actions=normalize_actions,
        depth_normalization_method=depth_normalization_method,
        action_normalization_method=action_normalization_method,
        normalize_action_indices=normalize_action_indices
    )

    # 2. Extract Stats from Train to pass to Val
    # This prevents Val from scanning the disk needlessly
    shared_stats = {
        'action_stats': train_dataset.action_stats,
        'obs_stats': train_dataset.obs_stats
    }

    # 3. Validation Dataset (Clean Data Only)
    print("Initializing Val Dataset (Using Train Stats)...")
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
        normalize_obs=normalize_obs,
        normalize_actions=normalize_actions,
        depth_normalization_method=depth_normalization_method,
        action_normalization_method=action_normalization_method,
        normalize_action_indices=normalize_action_indices,
        # OPTIMIZATION: Pass the stats here!
        precomputed_stats=shared_stats 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # Keep False for safety first run
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
    
    print("Dataloaders ready.")
    return train_loader, val_loader

def create_dataloaders_from_config(cfg) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg.data
    if cfg.is_regression:
        return create_wp_dataloaders(
        h5_file_path=data_cfg.h5_file_path,
        batch_size=data_cfg.batch_size,
        sequence_length=data_cfg.sequence_length,
        observation_mode=cfg.get('observation_mode', 'points'), 
        num_workers=data_cfg.num_workers,
    )
    else:
        return create_dataloaders(
            h5_file_path=data_cfg.h5_file_path,
            batch_size=data_cfg.batch_size,
            sequence_length=data_cfg.sequence_length,
            future_sequence_length=data_cfg.get('future_sequence_length', None),
            action_dim=data_cfg.action_dim,
            num_points=data_cfg.get('num_points', 0),
            train_split=data_cfg.train_split,
            normalize_obs=data_cfg.get('normalize_obs', True),
            normalize_actions=data_cfg.get('normalize_actions', True),
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
