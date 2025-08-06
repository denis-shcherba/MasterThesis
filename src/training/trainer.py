import wandb
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import logging

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

class Trainer:
    """
    Trainer class for policy learning.
    """
    
    def __init__(self, model, optimizer, criterion, device, cfg):
        """
        Initialize trainer.
        
        Args:
            model: Policy model
            optimizer: Optimizer
            criterion: Loss function
            device: Training device
            cfg: Configuration
        """
        self.model = model
        self.observation_mode = cfg.get('observation_mode', 'points')  # Default to 'points
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        
        # Create scheduler if specified
        self.scheduler = create_scheduler(cfg.get('scheduler', {}), optimizer)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(cfg.get('output_dir', 'outputs'))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if cfg.get('wandb', {}).get('enabled', False):
            self.init_wandb()
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb_cfg = self.cfg.wandb
        wandb.init(
            project=wandb_cfg.get('project', 'robot-manipulation'),
            name=wandb_cfg.get('name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=dict(self.cfg)
        )
        wandb.watch(self.model)
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        Args:
            train_loader: Training data loader
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Move batch to device
            obs = batch["observation"].to(self.device)  
            actions = batch['action'].to(self.device)  
            
            is_sequence = obs.shape[1] > 1  
    
            # Debug shapes
            tqdm.write(f"Obs shape: {obs.shape}, Actions shape: {actions.shape}, Is sequence: {is_sequence}")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare inputs
            state = None
            timestep = None
            
            if 'previous_actions' in batch and hasattr(self.model, 'state_dim') and self.model.state_dim > 0:
                state = batch['previous_actions'].to(self.device)
                
            if "timestep" in batch:
                timestep = batch["timestep"].to(self.device)
            
            # Forward pass based on policy type
            if self.model.policy_head_type == "mlp":
                output = self.model(obs, state=state, time_steps=timestep)
                
            elif self.model.policy_head_type == "gru":
                output = self.model(obs, state=state, hidden_state=None)
                
            elif self.model.policy_head_type == "transformer":
                if not is_sequence:
                    raise ValueError("Transformer requires sequence input but got single timestep")
                output = self.model(obs, state=state)
                
            else:
                output = self.model(obs)
            
            # Handle different output types and compute loss
            if isinstance(output, tuple):
                # GRU case: output is (actions, hidden_state)
                pred_actions, _ = output  # pred_actions shape: (B, T, A)
                
                if is_sequence:
                    # For sequences, we need to compute loss over all timesteps
                    # Reshape for loss computation: (B*T, A)
                    batch_size, seq_len = pred_actions.shape[:2]
                    pred_actions_flat = pred_actions.reshape(batch_size * seq_len, -1)
                    actions_flat = actions.reshape(batch_size * seq_len, -1)
                    loss = self.criterion(pred_actions_flat, actions_flat)
                else:
                    # Single timestep case
                    pred_actions = pred_actions.squeeze(1)  # Remove sequence dimension
                    loss = self.criterion(pred_actions, actions)
                    
            else:
                # MLP or Transformer case: output is just actions
                pred_actions = output
                
                if self.model.policy_head_type == "transformer":
                    # Transformer returns last timestep action: (B, A)
                    # Actions should be the target for the last timestep: (B, A) or (B, T, A)
                    if actions.dim() == 3:  # (B, T, A)
                        target_actions = actions[:, -1, :]  # Take last timestep
                    else:  # (B, A)
                        target_actions = actions
                    loss = self.criterion(pred_actions, target_actions)
                    
                elif self.model.policy_head_type == "mlp":
                    if is_sequence:
                        # MLP processes each timestep independently
                        # pred_actions shape: (B*T, A), actions shape: (B, T, A) or (B*T, A)
                        if actions.dim() == 3:  # (B, T, A)
                            batch_size, seq_len = actions.shape[:2]
                            actions_flat = actions.reshape(batch_size * seq_len, -1)
                        else:  # Already flat
                            actions_flat = actions
                        loss = self.criterion(pred_actions, actions_flat)
                    else:
                        # Single timestep
                        loss = self.criterion(pred_actions, actions)
            
            #Debug loss
            if num_batches % 10 == 0:  # tqdm.write every 10 batches
                tqdm.write(f"Batch {num_batches}: Loss = {loss.item():.6f}")
                tqdm.write(f"Pred actions range: [{pred_actions.min().item():.3f}, {pred_actions.max().item():.3f}]")
                tqdm.write(f"Target actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"WARNING: Invalid loss detected: {loss.item()}")
                tqdm.write(f"Pred actions stats: mean={pred_actions.mean():.3f}, std={pred_actions.std():.3f}")
                tqdm.write(f"Target actions stats: mean={actions.mean():.3f}, std={actions.std():.3f}")
                continue  # Skip this batch
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.cfg.get('train', {}).get('grad_clip', 0) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.get('train').get('grad_clip')
                )
                if num_batches % 10 == 0:
                    tqdm.write(f"Gradient norm: {grad_norm:.3f}")
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Avg': f'{total_loss/num_batches:.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                # Move batch to device
                obs = batch["observation"].to(self.device)
                actions = batch['action'].to(self.device)
                


                if 'previous_actions' in batch and hasattr(self.model, 'state_dim') and self.model.state_dim > 0:
                    state = batch['previous_actions'].to(self.device)
                    
                    if self.model.policy_head_type == "mlp":
                        timestep = batch["timestep"].to(self.device) 
                        output = self.model(obs, state, time_steps=timestep)
                    elif self.model.policy_head_type == "gru":
                        output = self.model(obs, state=state, hidden_state=None)
                    elif self.model.policy_head_type == "transformer":
                        timestep = batch["timestep"].to(self.device) 
                        output = self.model(obs, state=state)
                else:
                    output = self.model(obs)
                
                if isinstance(output, tuple):
                    pred_actions, _ = output
                    pred_actions = pred_actions.squeeze(1) 
                else:
                    pred_actions = output

                # Compute loss
                loss = self.criterion(pred_actions, actions)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
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
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            log.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        log.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            log.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log metrics
            log.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Log to wandb if enabled
            if self.cfg.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        log.info(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
