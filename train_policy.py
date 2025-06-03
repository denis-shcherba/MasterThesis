import hydra
from omegaconf import DictConfig
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_handling.dataset import create_dataloaders_from_config
from models.policy_head.policy_network import create_model
from training.trainer import Trainer
from training.losses import create_loss_function
from training.optimizer import create_optimizer

# Set up logging
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
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

        # Initialize model
        log.info("Initializing model...")
        model = create_model(cfg.model).to(device)
        log.info(f"Model created: {type(model).__name__}")
        log.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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