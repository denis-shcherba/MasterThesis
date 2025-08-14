#TODO fix prints for depth

import hydra
from omegaconf import DictConfig
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml

from data_handling.dataset import create_dataloaders_from_config
from models.policy_head.policy_network import create_model
from training.trainer import Trainer
from training.losses import create_loss_function
from training.optimizer import create_optimizer
from models.policy_head.policy_network import MultiModalPolicy, SimplePCToPosRegressor, PositionalEncoding
from utils.data_utils import numpy_to_python


log = logging.getLogger(__name__)


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

    # Set device
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    try:
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
            break
        
        # --- Access normalization stats from the train dataset ---
        train_dataset = train_loader.dataset
        action_stats = train_dataset.action_stats
        depth_stats = train_dataset.depth_stats

        # Convert to clean dicts
        normalization_stats = {
            "action_stats": numpy_to_python(action_stats) if action_stats else None,
            "depth_stats": numpy_to_python(depth_stats) if depth_stats else None,
        }

        output_dir = Path(cfg.get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "normalization_stats.yaml", "w") as f:
            yaml.dump(normalization_stats, f, default_flow_style=False)

        log.info(f"Saved clean normalization stats to {output_dir / 'normalization_stats.yaml'}")

        # Initialize model
        log.info("Initializing model...")
        # Assuming create_model returns either MultiModalPolicy or SimplifiedPolicy
        model = create_model(cfg.model).to(device)
        log.info(f"Model created: {type(model).__name__}")

        # --- Parameter Counting for Different Components ---
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters: {total_params:,}")

        pointnet_params = 0
        if hasattr(model, 'pointnet') and model.pointnet is not None:
            pointnet_params = sum(p.numel() for p in model.pointnet.parameters() if p.requires_grad)
            log.info(f"  - PointNet parameters: {pointnet_params:,} ({pointnet_params/total_params*100:.2f}%)")
        else:
            log.warning("Model does not have a 'pointnet' component, skipping PointNet parameter count.")


        policy_specific_params = 0
        component_params_sum = pointnet_params

        # Dynamically check the type of policy and count relevant parameters
        if isinstance(model, MultiModalPolicy):
            # Observation encoder (PointNet or DepthImageEncoder)
            if hasattr(model, 'obs_encoder') and model.obs_encoder is not None:
                obs_encoder_params = sum(p.numel() for p in model.obs_encoder.parameters() if p.requires_grad)
                log.info(f"  - Obs Encoder ({type(model.obs_encoder).__name__}) parameters: {obs_encoder_params:,} ({obs_encoder_params/total_params*100:.2f}%)")
                component_params_sum += obs_encoder_params

            # State encoder
            if hasattr(model, 'state_encoder') and model.state_encoder is not None:
                state_encoder_params = sum(p.numel() for p in model.state_encoder.parameters() if p.requires_grad)
                log.info(f"  - State Encoder parameters: {state_encoder_params:,} ({state_encoder_params/total_params*100:.2f}%)")
                component_params_sum += state_encoder_params

            # Time encoder
            if hasattr(model, 'time_encoder') and model.time_encoder is not None:
                if isinstance(model.time_encoder, (nn.Embedding, nn.Linear)):
                    time_encoder_params = sum(p.numel() for p in model.time_encoder.parameters() if p.requires_grad)
                    log.info(f"  - Time Encoder parameters: {time_encoder_params:,} ({time_encoder_params/total_params*100:.2f}%)")
                    component_params_sum += time_encoder_params
                elif isinstance(model.time_encoder, PositionalEncoding):
                    log.info("  - Time Encoder (PositionalEncoding) has no trainable parameters.")

            # Policy head
            if hasattr(model, 'policy_head') and model.policy_head is not None:
                policy_specific_params = sum(p.numel() for p in model.policy_head.parameters() if p.requires_grad)
                log.info(f"  - Policy Head ({type(model.policy_head).__name__}) parameters: {policy_specific_params:,} ({policy_specific_params/total_params*100:.2f}%)")
                component_params_sum += policy_specific_params


        elif isinstance(model, SimplePCToPosRegressor):
            # SimplifiedPolicy specific components
            if hasattr(model, 'regressor_head') and model.regressor_head is not None:
                policy_specific_params = sum(p.numel() for p in model.regressor_head.parameters() if p.requires_grad)
                log.info(f"  - Regressor Head parameters: {policy_specific_params:,} ({policy_specific_params/total_params*100:.2f}%)")
                component_params_sum += policy_specific_params
        
        else:
            log.warning("Unknown policy type encountered. Cannot provide detailed breakdown of policy-specific parameters.")

        # Final consistency check for total parameters
        # Allow for small discrepancies if any non-component params exist
        if abs(total_params - component_params_sum) > 10:
            log.warning(
                f"Parameter count mismatch ({total_params:,} vs {component_params_sum:,}). "
                "There might be other parameters in the main model not accounted for in specific components."
            )
        # Initialize optimizer
        log.info("Initializing optimizer...")
        optimizer = create_optimizer(cfg.optimizer, model.parameters())

        # Initialize loss function
        log.info("Initializing loss function...")
        criterion = create_loss_function(cfg.loss)

        # Initialize trainer
        log.info("Initializing trainer...")
        trainer = Trainer(model, optimizer, criterion, device, cfg)

        # Start training
        log.info("Starting training...")
        trainer.train(train_loader, val_loader, cfg.train.epochs)

        log.info("Training completed successfully!")

    except Exception as e:
        log.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    train_policy()