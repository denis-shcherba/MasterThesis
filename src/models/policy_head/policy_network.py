import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class PolicyNetwork(nn.Module):
    """
    Complete policy network that combines perception and action prediction.
    Designed for non-prehensile manipulation tasks.
    """
    
    def __init__(
        self,
        perception_model: nn.Module,
        policy_head: nn.Module,
        action_dim: int = 6,
        feature_dim: int = 256,
        dropout_rate: float = 0.1
    ):
        super(PolicyNetwork, self).__init__()
        
        self.perception_model = perception_model
        self.policy_head = policy_head
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Optional normalization layer
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, point_cloud: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete policy network.
        
        Args:
            point_cloud: Point cloud tensor of shape (batch_size, num_points, 3)
            **kwargs: Additional inputs if needed
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions (batch_size, action_dim)
                - 'features': Extracted features (batch_size, feature_dim)
        """
        # Extract features using perception model (PointNet)
        features = self.perception_model(point_cloud)
        
        # Apply normalization and dropout
        features = self.feature_norm(features)
        features = self.dropout(features)
        
        # Predict actions using policy head
        actions = self.policy_head(features)
        
        return {
            'actions': actions,
            'features': features
        }
    
    def get_action(self, point_cloud: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get action for inference (single step).
        
        Args:
            point_cloud: Point cloud tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action tensor
        """
        with torch.no_grad():
            output = self.forward(point_cloud)
            return output['actions']
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute policy loss.
        
        Args:
            predictions: Predicted actions (batch_size, action_dim)
            targets: Target actions from teacher (batch_size, action_dim)
            loss_type: Type of loss ('mse', 'huber', 'l1')
            
        Returns:
            Loss tensor
        """
        if loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        elif loss_type == 'huber':
            return F.huber_loss(predictions, targets)
        elif loss_type == 'l1':
            return F.l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def save_checkpoint(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'model_config': {
                'action_dim': self.action_dim,
                'feature_dim': self.feature_dim
            }
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0)