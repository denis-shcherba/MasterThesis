#TODO fix prints for depth

import hydra
from omegaconf import DictConfig
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
import socket

from data_handling.dataset import create_dataloaders_from_config
from utils.data_utils import  denormalize_actions

import robotic as ry
import matplotlib.pyplot as plt
import gymnasium as gym
import envs.env  # noqa: F401 
import time
log = logging.getLogger(__name__)


def test_depth_sequence(depth_batch):
    # take the first entry in the batch -> shape [10, 96, 96]
    for i in range(depth_batch.shape[0]):

        depth_seq = depth_batch[i]
        # create a figure with 2 rows x 4 cols for the 8 images
        fig, axes = plt.subplots(2, 4, figsize=(15, 6))

        for j, ax in enumerate(axes.flat):
            ax.imshow(depth_seq[j], cmap='viridis')  # use 'gray' if you prefer
            ax.set_title(f"Depth {j}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def test_pos_sequence(previous_action_seq_batch, target_action_seq_batch):
    env = gym.make("ShelfEnv-v0")
    obs, info = env.reset()
    

    with open("/home/denis/git/MasterThesis/outputs/final_outputs/normalization_stats_1000.yaml", 'r') as file:
        normalization_stats = yaml.safe_load(file)
    
    
    for i in range(target_action_seq_batch.shape[0]):
        target_action_seq = target_action_seq_batch[i]
        previous_action_seq = previous_action_seq_batch[i]
        target_action_seq = denormalize_actions(target_action_seq, normalization_stats["action_stats"])
        previous_action_seq = denormalize_actions(previous_action_seq, normalization_stats["action_stats"])

        for j in range(previous_action_seq.shape[0]):
            env.unwrapped.C.addFrame(f"prev_gripper{j}").setPosition(previous_action_seq[-(j+1)].cpu().numpy()).setShape(ry.ST.sphere, [.02]).setColor([.05*j,1-.05*j,1])
            env.unwrapped.C.getFrame("gripper").setPosition([0,0,0])
            print("Position:", previous_action_seq[-(j+1)].cpu().numpy())
            env.unwrapped.C.view(True)
            time.sleep(0.1)

        for j in range(target_action_seq.shape[0]):
            env.unwrapped.C.addFrame(f"gripper{j}").setPosition(target_action_seq[j].cpu().numpy()).setShape(ry.ST.sphere, [.02]).setColor([1,.5+.05*j,.5-.05*j])
            env.unwrapped.C.getFrame("gripper").setPosition([0,0,0])

            print("Position:", target_action_seq[j].cpu().numpy())
            env.unwrapped.C.view(True)
            time.sleep(0.1)

        for j in range(previous_action_seq.shape[0]):
            env.unwrapped.C.delFrame(f"prev_gripper{j}")
        for j in range(target_action_seq.shape[0]):
            env.unwrapped.C.delFrame(f"gripper{j}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_policy(cfg: DictConfig) -> None:
    """
    Main training function for manipulation policy learning.
    Args:
        cfg: Hydra configuration object
    """
    
    log.info("Starting policy training...")
    log.info(f"Experiment: {cfg.experiment_name}")
    log.info(f"Config: {cfg}")

    hostname = socket.gethostname()

    if hostname=="hal-9000":
        device_id = 0  # physical GPU ID seen free in nvidia-smi

        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    log.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create dataloaders
    log.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders_from_config(cfg)
    log.info(f"Dataset created successfully!")
    log.info(f"Train samples: {len(train_loader.dataset)}")
    log.info(f"Val samples: {len(val_loader.dataset)}")

    # Test loading a batch
    log.info("Testing batch loading...")
    for batch in train_loader:
        log.info(f"Batch shapes:")
        for key, value in batch.items():
            log.info(f"  {key}: {value.shape}")
        #test_pos_sequence(batch['previous_actions_sequence'], batch['target_actions_sequence'])  # Visualize pos sequence from the first batch
        test_depth_sequence(batch['observation_sequence'])  # Visualize depth sequence from the first batch
        break

if __name__ == "__main__":
    train_policy()