import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.perception.pointnet import PointNet
from models.policy_head.mlp_head import MLPHead, ResidualMLPHead

import inspect

class PointCloudPolicy(nn.Module):
    """
    Policy network that processes point clouds and outputs actions.
    Combines PointNet feature extraction with MLP policy head.
    """
    
    def __init__(self, num_points=1024, feature_dim=256, action_dim=6, 
                 hidden_dims=None, dropout_rate=0.3, use_residual=False,
                 output_activation=None):
        """
        Initialize PointCloud Policy network.
        
        Args:
            num_points (int): Number of points in input point cloud
            feature_dim (int): Dimension of PointNet features
            action_dim (int): Dimension of output actions
            hidden_dims (list): Hidden layer dimensions for policy head
            dropout_rate (float): Dropout rate for regularization
            use_residual (bool): Whether to use residual MLP head
            output_activation (str): Output activation function
        """
        super(PointCloudPolicy, self).__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # PointNet feature extractor
        self.pointnet = PointNet(
            num_points=num_points, 
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # Policy head
        if use_residual:
            self.policy_head = ResidualMLPHead(
                input_dim=feature_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                output_activation=output_activation
            )
        else:
            self.policy_head = MLPHead(
                input_dim=feature_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                output_activation=output_activation
            )
    
    def forward(self, point_cloud):
        """
        Forward pass through the policy network.
        
        Args:
            point_cloud: Point cloud tensor of shape (batch_size, num_points, 3)
            
        Returns:
            actions: Action tensor of shape (batch_size, action_dim)
        """
        # Extract features using PointNet
        features = self.pointnet(point_cloud)
        
        # Generate actions using policy head
        actions = self.policy_head(features)
        
        return actions
    
    def get_features(self, point_cloud):
        """
        Extract features from point cloud without generating actions.
        Useful for analysis and debugging.
        
        Args:
            point_cloud: Point cloud tensor of shape (batch_size, num_points, 3)
            
        Returns:
            features: Feature tensor of shape (batch_size, feature_dim)
        """
        return self.pointnet(point_cloud)


class MultiModalPolicy(nn.Module):
    """
    Policy network that can handle multiple input modalities.
    Currently supports point clouds and optional additional state information.
    """
    
    def __init__(self, num_points=1024, feature_dim=256, 
                 state_dim=0, action_dim=6, hidden_dims=None, 
                 dropout_rate=0.3, fusion_method='concat'):
        """
        Initialize MultiModal Policy network.
        
        Args:
            num_points (int): Number of points in input point cloud
            pointnet_feature_dim (int): Dimension of PointNet features
            state_dim (int): Dimension of additional state information
            action_dim (int): Dimension of output actions
            hidden_dims (list): Hidden layer dimensions for policy head
            dropout_rate (float): Dropout rate for regularization
            fusion_method (str): Method to fuse modalities ('concat', 'add')
        """
        super(MultiModalPolicy, self).__init__()
        
        self.num_points = num_points
        self.pointnet_feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fusion_method = fusion_method
        
        # PointNet feature extractor
        self.pointnet = PointNet(
            num_points=num_points,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # State encoder (if additional state is provided)
        self.state_encoder = None
        if state_dim > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, feature_dim)
            )
        
        # Determine input dimension for policy head
        if state_dim > 0:
            if fusion_method == 'concat':
                policy_input_dim = feature_dim + feature_dim
            elif fusion_method == 'add':
                policy_input_dim = feature_dim
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            policy_input_dim = feature_dim
        
        # Policy head
        self.policy_head = MLPHead(
            input_dim=policy_input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
    
    def forward(self, point_cloud, state=None):
        """
        Forward pass through the multimodal policy network.
        
        Args:
            point_cloud: Point cloud tensor of shape (batch_size, num_points, 3)
            state: Optional state tensor of shape (batch_size, state_dim)
            
        Returns:
            actions: Action tensor of shape (batch_size, action_dim)
        """
        # Extract point cloud features
        pc_features = self.pointnet(point_cloud)
        
        # Fuse with state if provided
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
            
            if self.fusion_method == 'concat':
                features = torch.cat([pc_features, state_features], dim=1)
            elif self.fusion_method == 'add':
                features = pc_features + state_features
        else:
            features = pc_features
        
        # Generate actions
        actions = self.policy_head(features)
        
        return actions


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def create_policy(policy_type='pointcloud', **kwargs):
    """
    Factory function to create different types of policies.
    Filters kwargs according to the policy constructor's expected arguments.
    """
    if policy_type == 'pointcloud':
        filtered = filter_kwargs(PointCloudPolicy.__init__, kwargs)
        return PointCloudPolicy(**filtered)
    elif policy_type == 'multimodal':
        filtered = filter_kwargs(MultiModalPolicy.__init__, kwargs)
        return MultiModalPolicy(**filtered)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
def create_model(model_cfg):
    """
    Create model based on configuration.
    
    Args:
        model_cfg: Model configuration from hydra config
        
    Returns:
        model: Initialized model
    """
    model = create_policy(
        policy_type=model_cfg.get('type', 'pointcloud'),
        num_points=model_cfg.get('num_points', 1024),
        feature_dim=model_cfg.get('feature_dim', 256),
        action_dim=model_cfg.get('action_dim', 7),
        hidden_dims=model_cfg.get('hidden_dims', [256, 128]),
        dropout_rate=model_cfg.get('dropout_rate', 0.3),
        use_residual=model_cfg.get('use_residual', False),
        output_activation=model_cfg.get('output_activation', None),
        # For multimodal policy
        state_dim=model_cfg.get('state_dim', 0),
        # fusion_method=model_cfg.get('fusion_method', 'concat'),
    )
    return model
