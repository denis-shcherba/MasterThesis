import wandb
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import logging
from abc import ABC, abstractmethod
import json
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from omegaconf import DictConfig
from diffusers.training_utils import EMAModel
# Set up logger
log = logging.getLogger(__name__)


def create_scheduler(scheduler_cfg, optimizer):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        scheduler_cfg: Scheduler configuration (dict)
        optimizer: Optimizer to schedule
        total_steps: Total number of training steps (needed for warmup schedulers)
        
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
    elif scheduler_type == 'cosine_warmup':
        total_steps = scheduler_cfg.get('total_steps', None)
        if total_steps is None:
            raise ValueError("total_steps must be provided for cosine_warmup scheduler")
        warmup_ratio = scheduler_cfg.get('warmup_ratio', 0.05)
        num_warmup_steps = int(warmup_ratio * scheduler_cfg.get('total_steps', 100))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

class BaseTrainer(ABC):
    """
    Abstract Base Class for training models. It handles the overall training loop,
    checkpointing, validation, and logging.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion, device, cfg: DictConfig):
        self.model = model.to(device)
        self.observation_mode = cfg.get('observation_mode', 'points')
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.scheduler = create_scheduler(cfg.get('scheduler', {}), optimizer) # (Assuming this function exists in your scope)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.output_dir = Path(cfg.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- EMA CHANGE: Initialize EMA Model ---
        self.use_ema = cfg.get('use_ema', False)
        if self.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=cfg.get('ema_decay', 0.9999),
                use_ema_warmup=True,
                power=cfg.get('ema_power', 0.75)
            )
            self.ema_model.to(self.device)
            # print("EMA initialized with decay 0.9999")
        # ----------------------------------------

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

    def _train_step(self, batch):
        """Performs a single training step for a batch."""
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        
        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write(f"WARNING: Invalid loss detected: {loss.item()}")
            return None 
        
        loss.backward()
        
        if self.cfg.get('train', {}).get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
        
        self.optimizer.step()

        # --- EMA CHANGE: Update EMA weights ---
        if self.use_ema:
            self.ema_model.step(self.model.parameters())
        # --------------------------------------

        return loss.item()

    def train_epoch(self, train_loader):
        """Iterates through the entire training dataset for one epoch."""
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            loss = self._train_step(batch)
            if loss is not None:
                total_loss += loss
                num_batches += 1
                pbar.set_postfix({'Loss': f'{loss:.6f}', 'Avg': f'{total_loss/num_batches:.6f}'})
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def validate_epoch(self, val_loader):
        """Runs one full validation pass on the validation dataset."""
        
        # --- EMA CHANGE: Swap weights for Validation ---
        # We want to validate the EMA model to see if it's actually better.
        if self.use_ema:
            self.ema_model.store(self.model.parameters()) # Save online weights
            self.ema_model.copy_to(self.model.parameters()) # Load EMA weights
        # -----------------------------------------------

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
        
        # --- EMA CHANGE: Restore Online weights ---
        # We must restore the noisy online weights so training can continue correctly next step.
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
        # ------------------------------------------

        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, counter_value, is_best=False, save_path=None):
            """
            Saves a checkpoint.
            """
            checkpoint = {
                'counter': counter_value,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'config': dict(self.cfg)
            }
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            # --- EMA CHANGE: Save EMA State ---
            if self.use_ema:
                checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            # ----------------------------------

            # Always save the latest checkpoint
            latest_path = self.output_dir / 'latest_checkpoint.pth'
            torch.save(checkpoint, latest_path)

            # Save the best checkpoint if applicable
            if is_best:
                best_path = self.output_dir / 'best_checkpoint.pth'
                torch.save(checkpoint, best_path)
                # log.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")

            # Save to a custom path if one was provided
            if save_path:
                torch.save(checkpoint, save_path)
    
    def _run_training_loop(self, train_loader, val_loader):
        is_step_based = 'total_steps' in self.cfg.trainer
        total_steps = float('inf')
        max_epochs = float('inf')
        
        checkpoint_interval = self.cfg.trainer.get('checkpoint_interval_steps', 
                                                   self.cfg.trainer.get('eval_interval_steps', float('inf')))
        if is_step_based:
            total_steps = self.cfg.trainer.total_steps
            if self.cfg.trainer.get('max_epochs') is not None:
                max_epochs = self.cfg.trainer.max_epochs
            log_interval = self.cfg.trainer.get('log_interval_steps', 1000)
            eval_interval = self.cfg.trainer.get('eval_interval_steps', 5000)
        else:
            max_epochs = self.cfg.trainer.epochs
            log_interval = self.cfg.trainer.get('log_interval_epochs', 1)
            eval_interval = self.cfg.trainer.get('eval_interval_epochs', 5)

        steps_per_epoch = len(train_loader)
        steps_done = 0
        epoch = 0

        log.info(f"Starting training...")
        if is_step_based:
            log.info(f"Targeting {total_steps} steps (~{total_steps / steps_per_epoch:.2f} epochs).")
            pbar = tqdm(total=total_steps, desc="Training Steps")
        else:
            log.info(f"Targeting {max_epochs} epochs (~{max_epochs * steps_per_epoch} steps).")

        try:
            while steps_done < total_steps and epoch < max_epochs:
                if not is_step_based:
                    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
                    
                batch_iterator = train_loader if is_step_based else pbar
                
                epoch_losses = []
                
                for batch in batch_iterator:
                    if steps_done >= total_steps:
                        break
                        
                    loss_value = self._train_step(batch)
                    
                    if loss_value is not None:
                        self.train_losses.append(loss_value)
                        epoch_losses.append(loss_value)
                        steps_done += 1

                        # --- ADD THIS BLOCK ---
                        # Step the scheduler here if it's step-based.
                        # Exclude ReduceLROnPlateau, which is handled after validation.
                        if is_step_based and self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step()
                        # ----------------------

                        if is_step_based:
                            avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                            pbar.set_postfix({
                                'Loss': f'{loss_value:.6f}', 
                                'Avg': f'{avg_loss:.6f}',
                                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                'Epoch': f'{epoch + 1}',
                            })
                            pbar.update(1)

                            if steps_done % checkpoint_interval == 0:
                                periodic_path = self.output_dir / f'step_{steps_done}_checkpoint.pth'
                                self.save_checkpoint(steps_done, is_best=False, save_path=periodic_path)
                                log.info(f"Saved periodic checkpoint at step {steps_done}")
                        else:
                            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                            pbar.set_postfix({'Loss': f'{loss_value:.6f}', 'Avg': f'{avg_epoch_loss:.6f}'})

                        if is_step_based:
                            log_data = {}
                            if steps_done % log_interval == 0:
                                log_data['train_loss'] = loss_value

                            if steps_done % eval_interval == 0:
                                pbar.clear()
                                val_loss = self.validate_epoch(val_loader)
                                pbar.refresh()
                                
                                self.val_losses.append(val_loss)
                                is_best = val_loss < self.best_val_loss
                                if is_best:
                                    self.best_val_loss = val_loss
                                
                                self.save_checkpoint(steps_done, is_best)
                                log.info(f"Step {steps_done}: Train Loss: {loss_value:.6f}, Val Loss: {val_loss:.6f}")
                                
                                log_data['val_loss'] = val_loss
                                log_data['learning_rate'] = self.optimizer.param_groups[0]['lr']

                            if log_data and self.cfg.get('wandb', {}).get('enabled', False):
                                wandb.log(log_data, step=steps_done)

                # End of epoch processing
                epoch += 1
                
                if not is_step_based:
                    if hasattr(pbar, 'close'):
                        pbar.close()
                        
                    # The epoch-based scheduler step is correctly placed here.
                    if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                    
                    val_loss = self.validate_epoch(val_loader)
                    # For ReduceLROnPlateau, step with the validation loss
                    if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)

                    self.val_losses.append(val_loss)
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    self.save_checkpoint(epoch, is_best)
                    epoch_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
                    log.info(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    if self.cfg.get('wandb', {}).get('enabled', False):
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': epoch_train_loss,
                            'val_loss': val_loss,
                            'learning_rate': self.optimizer.param_groups[0]['lr']
                        })

        finally:
            if is_step_based and 'pbar' in locals():
                pbar.close()

        log.info(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        
        curves = {"train_losses": self.train_losses, "val_losses": self.val_losses}
        curves_path = self.output_dir / "training_curves.json"
        with open(curves_path, "w") as f:
            json.dump(curves, f, indent=4)
        log.info(f"Saved training curves to {curves_path}")

    def train(self, train_loader, val_loader):
        """Public entry point for training."""
        self._run_training_loop(train_loader, val_loader)



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

    def move_to_device(self, batch):
        # This function handles the logic of moving tensors to the specified device
        if isinstance(batch, (list, tuple)):
            # If the batch is a list or tuple of tensors, iterate through them
            return [self._move_to_device_tensor(item) for item in batch]
        elif isinstance(batch, dict):
            # If the batch is a dictionary of tensors, move each value
            return {key: self._move_to_device_tensor(value) for key, value in batch.items()}
        else:
            # Handle the case where the batch is a single tensor
            return self._move_to_device_tensor(batch)

    def _move_to_device_tensor(self, tensor):
        # The actual tensor-moving logic
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor


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

class RegressionPolicyTrainer(BaseTrainer):
    def _compute_loss(self, batch):
        # Shape: (batch_size, num_points, 3)
        initial_obs = batch["initial_observation"].to(self.device)
        if initial_obs.dim() == 3: 
             # Ensure correct dimensions if your dataloader squeezes/unsqueezes differently
             pass 

        target_action = batch["waypoints"].to(self.device)[:, :, :]
        
        # Check shapes: target_action should be (batch_size, 2, 3)
        
        # Forward pass
        pred_actions = self.model(initial_obs) 
        # pred_actions shape is now also (batch_size, 2, 3) thanks to the model update

        loss = self.criterion(pred_actions, target_action)
        return loss

class DiffusionPolicyTrainer(BaseTrainer):
    """
    Trainer for a diffusion policy.
    The model is expected to return a dictionary with 'noise_pred' and 'noise_target'
    during training, and the criterion should be a simple MSE loss.
    """
    def _compute_loss(self, batch):
        # 1. Move batch to the correct device
        # Note: Your dataloader should provide observation and target action sequences
        batch = self.move_to_device(batch)
        obs_seq = batch["observation_sequence"]
        state_seq = batch.get("previous_actions_sequence") # Assuming you might have this
        target_actions_seq = batch["target_actions_sequence"]
        
        # 2. Forward pass through the model
        # During training, the model's forward pass requires the `true_actions`
        # and will return the dictionary we need.
        outputs = self.model(
            observations=obs_seq, 
            states=state_seq, 
            true_actions=target_actions_seq
        )
        
        # 3. Compute loss using the prediction and target from the model's output
        # Your self.criterion should be nn.MSELoss for this trainer.
        loss = self.criterion(outputs['noise_pred'], outputs['noise_target'])
        
        return loss

    def move_to_device(self, batch):
        # You can reuse or inherit this helper method
        if isinstance(batch, dict):
            return {key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        return batch # Or handle other types if necessary
    

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

    elif cfg.model.get("policy_head_type") == 'diffusion':
        return DiffusionPolicyTrainer(model, optimizer, criterion, device, cfg)

    elif cfg.model.get("type") == "regression":
        return RegressionPolicyTrainer(model, optimizer, criterion, device, cfg)

    else:
        return SequencePolicyTrainer(model, optimizer, criterion, device, cfg)
