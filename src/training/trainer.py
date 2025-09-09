import wandb
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import logging
from abc import ABC, abstractmethod
import json

# Set up logger
log = logging.getLogger(__name__)


def create_scheduler(scheduler_cfg, optimizer):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        scheduler_cfg: Scheduler configuration
        optimizer: Optimizer to schedule
        
    Returns:
        scheduler: Initialized scheduler or None
    """
    if not scheduler_cfg or not scheduler_cfg.get('use', False):
        return None
    
    scheduler_type = scheduler_cfg.get('type', 'reduce_on_plateau').lower()
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_cfg.get('factor', 0.5),
            patience=scheduler_cfg.get('patience', 10),
            verbose=True
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_cfg.get('step_size', 30),
            gamma=scheduler_cfg.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get('T_max', 100),
            eta_min=scheduler_cfg.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

class BaseTrainer(ABC):
    """
    Abstract Base Class for training models. It handles the overall training loop,
    checkpointing, validation, and logging.
    
    Subclasses must implement the _compute_loss method.
    """
    def __init__(self, model, optimizer, criterion, device, cfg):
        self.model = model
        self.observation_mode = cfg.get('observation_mode', 'points')
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        # self.scheduler = create_scheduler(cfg.get('scheduler', {}), optimizer)
        self.scheduler = None # Placeholder
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.output_dir = Path(cfg.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get('wandb', {}).get('enabled', False):
            self.init_wandb()

    def init_wandb(self):
        wandb_cfg = self.cfg.wandb
        wandb.init(
            project=wandb_cfg.get('project', 'robot-manipulation'),
            name=wandb_cfg.get('name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=dict(self.cfg)
        )
        wandb.watch(self.model)

    @abstractmethod
    def _compute_loss(self, batch):
        pass

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"WARNING: Invalid loss detected: {loss.item()}")
                continue
            loss.backward()
            if self.cfg.get('train', {}).get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Avg': f'{total_loss/num_batches:.6f}'})
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': dict(self.cfg)
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            log.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")

    def train(self, train_loader, val_loader, num_epochs):
        log.info(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            log.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            if (epoch + 1) % self.cfg.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            log.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if self.cfg.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        log.info(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        curves = {"train_losses": self.train_losses, "val_losses": self.val_losses}
        curves_path = self.output_dir / "training_curves.json"
        with open(curves_path, "w") as f:
            json.dump(curves, f, indent=4)
        log.info(f"Saved training curves to {curves_path}")


class SequencePolicyTrainer(BaseTrainer):
    """
    Trainer for sequence-to-sequence models that predict an entire action sequence.
    """
    def _compute_loss(self, batch):
        # 1. Get the sequences from the batch (from your new dataset)
        obs_seq = batch["observation_sequence"].to(self.device)
        prev_actions_seq = batch['previous_actions_sequence'].to(self.device)
        target_actions_seq = batch['target_actions_sequence'].to(self.device)
        
        # 2. Forward pass with both observation and previous action sequences
        predicted_actions_seq = self.model(obs_seq, prev_actions_seq)
        
        # 3. Compute loss between the predicted sequence and the target sequence
        # The criterion (e.g., MSELoss) will automatically handle the sequence dimension.
        loss = self.criterion(predicted_actions_seq, target_actions_seq)
        
        return loss

class ActionPolicyTrainer(BaseTrainer):
    """
    Trainer for models that predict a SINGLE action vector from a history.
    """
    def _compute_loss(self, batch):
        # This trainer expects a history of observations and a SINGLE target action.
        # Your original dataset was structured this way.
        obs_history = batch["observation_sequence"].to(self.device)
        action_history = batch['previous_actions_sequence'].to(self.device) # This is the state/history
        target_action = batch['target_actions_sequence'].to(self.device) # The single target action
        
        # --- Forward pass ---
        # The model takes the history of observations and past actions
        pred_actions = self.model(obs_history, action_history) 
        
        # For a single-action prediction model, we might only care about the last output
        if pred_actions.dim() == 3: # If model outputs a sequence, take the last step
            pred_actions = pred_actions[:, -1, :]

        # --- Loss computation ---
        loss = self.criterion(pred_actions, target_action)
        return loss


from tqdm import tqdm

class WaypointTimingTrainer(BaseTrainer):
    """
    Trainer for models that predict both waypoints and timings.
    """
    def _compute_loss(self, batch):
        # --- Prepare Data ---
        obs = batch["observation"].to(self.device)
        first_obs = batch["first_obs"].to(self.device) 
        state = batch['previous_actions'].to(self.device)
        actions = batch["action"].to(self.device)
        waypoints_gt = batch["waypoints"].to(self.device)

        # --- Forward Pass ---
        output = self.model(obs, state, first_obs)

        # --- Prepare Predictions and Targets for Loss Module ---
        predictions = {
            "timing": output["timings"],
            "waypoint": output["waypoints"] # Pass the list of tensors directly
        }

        targets = {
            "timing": actions[:, -1, :] if actions.dim() == 3 else actions,
            "waypoint": waypoints_gt
        }

        # --- Calculate Loss ---
        # A single call to the criterion, which is now our MultiLoss module.
        # It handles the weighting and sub-loss calculations internally.
        loss = self.criterion(predictions, targets)
        
        # Optional: You can still log individual components if needed,
        # but the main calculation is now encapsulated.

        return loss
    
    
# Factory function to create appropriate trainer
def create_trainer(model, optimizer, criterion, device, cfg):
    """
    Factory function to create the appropriate trainer.
    
    Args:
        trainer_type: 'policy' or 'waypoint_timing'
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        cfg: Configuration
        
    Returns:
        Appropriate trainer instance
    """
    if cfg.get("is_waypointPlusTimings", False):
        return WaypointTimingTrainer(model, optimizer, criterion, device, cfg)

    else:
        return SequencePolicyTrainer(model, optimizer, criterion, device, cfg)
