import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WaypointDataset(Dataset):
    """
    Minimal HDF5 dataset loader:
    - loads only observations
    - loads optional waypoints
    - loads optional book_params
    - no actions, no paths, no normalization
    - supports sliding window sequences
    """

    def __init__(
        self,
        h5_file_path: str,
        sequence_length: int = 1,
        split: str = 'train',
        train_split: float = 0.8,
        observation_mode: str = 'depth',
        normalize_depth: bool = True,
        depth_normalization_method: str = 'minmax',
        random_seed: int = 42,
    ):
        self.h5_file_path = h5_file_path
        self.sequence_length = sequence_length
        self.split = split
        self.normalize_depth = normalize_depth
        self.observation_mode = observation_mode
        self.depth_normalization_method = depth_normalization_method
        self.rng = np.random.default_rng(random_seed)

        # Determine observation key
        if observation_mode == 'depth':
            self.obs_key = 'depth'
        elif observation_mode in ['points', 'sam_points']:
            self.obs_key = 'points'
            self.normalize_depth = False
        elif observation_mode == 'dino_cls':
            self.obs_key = 'cls_features'
            self.normalize_depth = False
        elif observation_mode == 'dino_patches':
            self.obs_key = 'patch_features'
            self.normalize_depth = False
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

        self.demo_meta = []
        self.valid_indices = []

        self._index_demonstrations()
        if split in ['train', 'val']:
            self._create_split(train_split)

        # Depth normalization stats
        if self.normalize_depth and self.obs_key == 'depth':
            with h5py.File(self.h5_file_path, 'r') as f:
                self.depth_stats = self._compute_depth_normalization_stats(f)
        else:
            self.depth_stats = None

        # preload to RAM
        print(f"Preloading {split} split...")
        self._preload_all_data()
        print("Preload complete.")

    # ----------------------------------------------------------
    # Index demos
    # ----------------------------------------------------------
    def _index_demonstrations(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            demo_keys = sorted([k for k in f if k.startswith("demo_")],
                               key=lambda x: int(x.split("_")[1]))

            for demo_idx, demo_key in enumerate(demo_keys):

                obs_key_path = f"{demo_key}/{self.obs_key}"
                if obs_key_path not in f:
                    continue

                obs_shape = f[obs_key_path].shape  # (T, ...)
                num_timesteps = obs_shape[0]
                obs_sample_shape = obs_shape[1:]

                self.demo_meta.append({
                    "demo_id": demo_idx,
                    "demo_key": demo_key,
                    "num_timesteps": num_timesteps,
                    "obs_sample_shape": obs_sample_shape,
                })

                # Valid windows
                if num_timesteps >= self.sequence_length:
                    for t in range(num_timesteps - self.sequence_length + 1):
                        self.valid_indices.append((demo_idx, t))

        if not self.demo_meta:
            raise ValueError("No valid demos found.")

    # ----------------------------------------------------------
    # Train/val split
    # ----------------------------------------------------------
    def _create_split(self, train_split: float):
        num = len(self.demo_meta)
        ids = np.arange(num)
        self.rng.shuffle(ids)
        split_idx = int(num * train_split)

        if self.split == "train":
            selected = set(ids[:split_idx])
        else:
            selected = set(ids[split_idx:])

        # filter demos
        self.demo_meta = [m for i, m in enumerate(self.demo_meta) if i in selected]

        # filter indices
        self.valid_indices = [(d, t) for (d, t) in self.valid_indices if d in selected]

        # remap demo IDs
        new_id_map = {old['demo_id']: i for i, old in enumerate(self.demo_meta)}
        for m in self.demo_meta:
            m["demo_id"] = new_id_map[m["demo_id"]]
        self.valid_indices = [(new_id_map[d], t) for (d, t) in self.valid_indices]

    # ----------------------------------------------------------
    # Depth stats
    # ----------------------------------------------------------
    def _compute_depth_normalization_stats(self, f):
        mins, maxs = [], []
        for meta in self.demo_meta:
            arr = f[f"{meta['demo_key']}/depth"][...]
            flat = arr.reshape(-1)
            mins.append(flat.min())
            maxs.append(flat.max())

        mn = float(np.min(mins))
        mx = float(np.max(maxs))
        return {"min": mn, "range": max(mx - mn, 1e-8)}

    # ----------------------------------------------------------
    # Preload everything
    # ----------------------------------------------------------
    def _preload_all_data(self):
        self.obs_cache = {}
        self.waypoint_cache = {}
        self.book_params_cache = {}
        self.initial_obs_cache = {}

        with h5py.File(self.h5_file_path, 'r') as f:
            for meta in self.demo_meta:
                k = meta["demo_key"]

                # observations
                self.obs_cache[k] = f[f"{k}/{self.obs_key}"][...].astype(np.float32)

                # waypoints
                if "waypoints" in f[k]:
                    self.waypoint_cache[k] = f[f"{k}/waypoints"][...].astype(np.float32)
                else:
                    self.waypoint_cache[k] = np.zeros((1,), dtype=np.float32)

                # book params
                if "book_params" in f[k]:
                    self.book_params_cache[k] = f[f"{k}/book_params"][...].astype(np.float32)
                else:
                    self.book_params_cache[k] = np.zeros((1,), dtype=np.float32)

                # initial observation
                self.initial_obs_cache[k] = self.obs_cache[k][0]

    # ----------------------------------------------------------
    # Get item
    # ----------------------------------------------------------
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        demo_idx, start_t = self.valid_indices[idx]
        meta = self.demo_meta[demo_idx]
        k = meta["demo_key"]

        end_t = start_t + self.sequence_length
        obs = self.obs_cache[k][start_t:end_t]

        if self.normalize_depth and self.obs_key == "depth":
            stats = self.depth_stats
            obs = (obs - stats["min"]) / stats["range"]

        return {
            "observations": torch.from_numpy(obs).float(),
            "waypoints": torch.from_numpy(self.waypoint_cache[k]).float(),
            "book_params": torch.from_numpy(self.book_params_cache[k]).float(),
            "initial_observation": torch.from_numpy(self.initial_obs_cache[k]).float(),
            "demo_id": torch.tensor(meta["demo_id"]).long(),
        }
    

def create_wp_dataloaders(
    h5_file_path: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    train_split: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42,
    observation_mode: str = 'points',
    normalize_depth: bool = True,
    depth_normalization_method='minmax'
):
    train_dataset = WaypointDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        split='train',
        train_split=train_split,
        observation_mode=observation_mode,
        normalize_depth=normalize_depth,
        depth_normalization_method=depth_normalization_method,
        random_seed=random_seed,
    )

    val_dataset = WaypointDataset(
        h5_file_path=h5_file_path,
        sequence_length=sequence_length,
        split='val',
        train_split=train_split,
        observation_mode=observation_mode,
        normalize_depth=normalize_depth,
        depth_normalization_method=depth_normalization_method,
        random_seed=random_seed,
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


def create_wp_dataloaders_from_config(cfg):
    data_cfg = cfg.data
    return create_wp_dataloaders(
        h5_file_path=data_cfg.h5_file_path,
        batch_size=data_cfg.batch_size,
        sequence_length=data_cfg.sequence_length,
        train_split=data_cfg.train_split,
        random_seed=data_cfg.random_seed,
        observation_mode=data_cfg.get('observation_mode', 'points'),
        normalize_depth=data_cfg.get('normalize_depth', True),
        depth_normalization_method=data_cfg.get('depth_normalization_method', 'minmax'),
        num_workers=data_cfg.num_workers,
    )
