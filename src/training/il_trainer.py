# prolly omit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging
from collections import defaultdict

from ..models.policy_head.policy_network import PolicyNetwork
from .losses import compute_policy_loss
from ..training.metrics import compute_metrics


class ImitationLearningTrainer:
    """
    Trainer for student policy using imitation learning.
    Supports standard behavioral cloning and DAgger training.
    """
    
    def __init__(
        self,
        student_policy: PolicyNetwork,
        teacher_policy: Optional[Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'cosine',
        loss_type: str = 'mse',
        grad_clip_norm: Optional[float] = 1.0,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        use_wandb: bool = False,
        wandb_project: str = 'manipulation_il'
    ):
        self.student_policy = student_policy.to(device)
        self.teacher_policy = teacher_policy
        self.device = device
        self.loss_type = loss_type
        self.grad_clip_norm = grad_clip_norm
        
        # Logging and checkpointing
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project=wandb_project, config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'scheduler_type': scheduler_type,
                'loss_type': loss_type,
                'grad_clip_norm': grad_clip_norm
            })
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.student_policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = self._get_scheduler(scheduler_type)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
    
    def _get_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student_policy.train()
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            point_clouds = batch['point_cloud'].to(self.device)
            target_actions = batch['action'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.student_policy(point_clouds)
            predicted_actions = outputs['actions']
            
            # Compute loss
            loss = compute_policy_loss(
                predicted_actions, 
                target_actions, 
                loss_type=self.loss_type
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.student_policy.parameters(), 
                    self.grad_clip_norm
                )
            
            self.optimizer.step()
            
            # Compute metrics
            batch_metrics = compute_metrics(predicted_actions, target_actions)
            batch_metrics['loss'] = loss.item()
            
            # Update epoch metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{batch_metrics.get('mse', 0):.4f}"
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    f'train/{key}': value for key, value in batch_metrics.items()
                }, step=self.global_step)
            
            self.global_step += 1
        
        # Average metrics over epoch
        epoch_avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        return epoch_avg_metrics
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.student_policy.eval()
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation {epoch}'):
                # Move batch to device
                point_clouds = batch['point_cloud'].to(self.device)
                target_actions = batch['action'].to(self.device)
                
                # Forward pass
                outputs = self.student_policy(point_clouds)
                predicted_actions = outputs['actions']
                
                # Compute loss
                loss = compute_policy_loss(
                    predicted_actions, 
                    target_actions, 
                    loss_type=self.loss_type
                )
                
                # Compute metrics
                batch_metrics = compute_metrics(predicted_actions, target_actions)
                batch_metrics['loss'] = loss.item()
                
                # Update epoch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)
        
        # Average metrics over epoch
        epoch_avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        return epoch_avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_every: int = 10,
        validate_every: int = 1
    ):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = {}
            if val_loader is not None and epoch % validate_every == 0:
                val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train MSE: {train_metrics.get('mse', 0):.4f}"
            )
            
            if val_metrics:
                self.logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val MSE: {val_metrics.get('mse', 0):.4f}"
                )
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
                if val_metrics:
                    log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict, step=epoch)
            
            # Save training history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch)
            
            # Save best model
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': dict(self.training_history)
        }
        
        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save best checkpoint
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.student_policy.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def evaluate_teacher_student_agreement(
        self,
        eval_loader: DataLoader,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate agreement between teacher and student policies.
        Useful for DAgger training.
        """
        if self.teacher_policy is None:
            raise ValueError("Teacher policy not provided")
        
        self.student_policy.eval()
        agreements = []
        action_diffs = []
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if samples_processed >= num_samples:
                    break
                
                point_clouds = batch['point_cloud'].to(self.device)
                batch_size = point_clouds.shape[0]
                
                # Get student predictions
                student_outputs = self.student_policy(point_clouds)
                student_actions = student_outputs['actions']
                
                # Get teacher predictions (assuming teacher can process point clouds)
                # You may need to adapt this based on your teacher policy interface
                teacher_actions = self.teacher_policy.get_action(point_clouds)
                
                # Compute agreement metrics
                action_diff = torch.norm(student_actions - teacher_actions, dim=1)
                action_diffs.extend(action_diff.cpu().numpy())
                
                # Compute agreement (actions within threshold are considered agreeing)
                threshold = 0.1  # Adjust based on your action space
                agreement = (action_diff < threshold).float()
                agreements.extend(agreement.cpu().numpy())
                
                samples_processed += batch_size
        
        return {
            'agreement_rate': np.mean(agreements),
            'mean_action_diff': np.mean(action_diffs),
            'std_action_diff': np.std(action_diffs)
        }