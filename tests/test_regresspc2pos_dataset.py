import hydra
from omegaconf import DictConfig
import torch
from envs.create_env import ShelfPullDataCollector 
import logging
import h5py


log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main evaluation/inference function for the manipulation policy.
    Args:
        cfg: Hydra configuration object.
    """

    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")
    # log.info(f"Full config: {OmegaConf.to_yaml(cfg)}") # For debugging

    # --- 1. Setup Device and Environment---
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = cfg.get("inference", {}).get("device", default_device_str)
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    collector = ShelfPullDataCollector(**cfg.env)
    collector.C.view(True)
    collector.C.view(False) 


    file_path = "variable_demo.h5" 

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Opening HDF5 file: {file_path}\n")

            # Get all top-level keys (groups/datasets)
            top_level_keys = list(f.keys()) # Convert to list to avoid issues if file changes during iteration

            # Filter for keys that start with 'demo_' and are groups
            demo_folders = sorted([key for key in top_level_keys if key.startswith("demo_") and isinstance(f[key], h5py.Group)])

            if not demo_folders:
                print("No 'demo_idx' folders (groups) found in the HDF5 file.")
            else:
                for folder_name in demo_folders:
                    print(f"--- Processing folder (group): /{folder_name}/ ---")
                    
                    # Access the group (folder)
                    demo_group = f[folder_name]

                    # Check if the 'path' dataset exists within this group
                    if 'path' in demo_group and isinstance(demo_group['path'], h5py.Dataset):
                        path_dataset = demo_group['path']
                        points = demo_group['points']


                        print(f"  Found 'path' dataset (3D array) with shape: {path_dataset.shape}")
                        
                        collector.C.getFrame("cameraStatic").setPointCloud(points)
                        collector.C.setJointState([path_dataset[0], path_dataset[1], path_dataset[2], 1, 0, 0, 0])  # Assuming the first 7 values are joint angles
                        
                        # For printing, we'll load the whole thing into memory
                        data = path_dataset[()] 
                        
                        collector.C.view(True)

                        print("  Contents of 'path' array:")
                        print(data)


                    else:
                        print(f"  'path' dataset not found or is not a dataset within /{folder_name}/")
                    print("-" * 40) # Separator for readability

    except FileNotFoundError:
        print(f"Error: HDF5 file not found at '{file_path}'. Please ensure the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()