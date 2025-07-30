#TODO hard

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
import gymnasium as gym
from envs.create_env import ShelfPullDataCollector 
from models.policy_head.policy_network import create_model
from data_handling.processing import normalize_point_cloud_to_unit_sphere_torch, pose_9d_to_7d, pose_7d_to_9d
import envs.env  # noqa: F401 

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="inference", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    """Main evaluation/inference function for the manipulation policy."""
    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")

    # Setup device
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = cfg.get("inference", {}).get("device", default_device_str)
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    # Initialize environment
    collector = ShelfPullDataCollector(**cfg.env)
    collector.spawn_books_scene()
    collector.C.view(False) 
    q0 = collector.C.getJointState()

    env = gym.make("ShelfEnv-v0")  # Adjust args as needed
    env.reset()
    env.render()
    env.step(env.action_space.sample())


    # Initialize model
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    
    log.info("Policy evaluation/inference finished")

if __name__ == "__main__":
    eval_policy()