import torch
import torch.nn as nn

from models.perception.pointnet import PointNet
from models.policy_head.mlp_head import MLPHead, GRUHead
import math
import inspect
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Generates sinusoidal positional encodings for timesteps."""
    def __init__(self, embedding_dim, max_timesteps=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        pe = torch.zeros(max_timesteps, embedding_dim)
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, time_steps):
        """Args: time_steps: Tensor of shape (batch_size,)"""
        return self.pe[time_steps, :]

class MultiModalPolicy(nn.Module):
    """
    Policy network that handles multiple input modalities and supports both
    stateless (MLP) and stateful (GRU) policy heads.
    """
    
    def __init__(self, num_points=1024, feature_dim=256, 
                 state_dim=0, action_dim=6, 
                 dropout_rate=0.3, fusion_method='concat',
                 time_encoding='none', time_embedding_dim=128, max_timesteps=64,
                 # --- New arguments for selecting and configuring the policy head ---
                 policy_head_type: str = 'mlp', 
                 mlp_hidden_dims: Optional[list] = None,
                 gru_hidden_dim: int = 256, 
                 gru_num_layers: int = 2):
        """
        Initialize MultiModal Policy network.
        
        Args:
            (Same as before, with new arguments below)
            policy_head_type (str): The type of policy head to use ('mlp' or 'gru').
            mlp_hidden_dims (list): Hidden layer dimensions for the MLP head.
            gru_hidden_dim (int): Hidden state dimension for the GRU head.
            gru_num_layers (int): Number of layers for the GRU head.
        """
        super(MultiModalPolicy, self).__init__()
        
        # --- Store configuration ---
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fusion_method = fusion_method
        self.time_encoding = time_encoding
        self.max_timesteps = max_timesteps
        self.policy_head_type = policy_head_type
        
        # --- PointNet feature extractor ---
        self.pointnet = PointNet(num_points=num_points, feature_dim=feature_dim, dropout_rate=dropout_rate)
        
        # --- State encoder ---
        self.state_encoder = nn.Linear(state_dim, feature_dim) if state_dim > 0 else None
        
        # --- Time encoder ---
        self.time_encoder = self._create_time_encoder(time_encoding, time_embedding_dim, max_timesteps)
        
        # --- Determine the input dimension for the policy head ---
        policy_input_dim = feature_dim  # Base features from PointNet
        if state_dim > 0 and fusion_method == 'concat':
            policy_input_dim += feature_dim
        if time_encoding != 'none':
            policy_input_dim += time_embedding_dim
        
        # --- Create the selected policy head ---
        if policy_head_type == 'mlp':
            if mlp_hidden_dims is None: mlp_hidden_dims = [256, 128]
            self.policy_head = MLPHead(
                input_dim=policy_input_dim, output_dim=action_dim, 
                hidden_dims=mlp_hidden_dims, dropout_rate=dropout_rate
            )
        elif policy_head_type == 'gru':
            self.policy_head = GRUHead(
                input_dim=policy_input_dim, output_dim=action_dim,
                hidden_dim=gru_hidden_dim, num_layers=gru_num_layers,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown policy_head_type: {policy_head_type}")

    def _create_time_encoder(self, encoding_type, embedding_dim, max_steps):
        if encoding_type == 'none': return None
        if encoding_type == 'positional': return PositionalEncoding(embedding_dim, max_steps)
        if encoding_type == 'embedding': return nn.Embedding(max_steps, embedding_dim)
        if encoding_type == 'linear': return nn.Linear(1, embedding_dim)
        raise ValueError(f"Unknown time_encoding method: {encoding_type}")
    
    def forward(self, point_cloud: torch.Tensor, state: Optional[torch.Tensor] = None, 
                time_steps: Optional[torch.Tensor] = None, 
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy. Handles both single-step and sequence data.

        Args:
            point_cloud: Point cloud tensor.
                         - For sequences: (batch_size, seq_len, num_points, 3)
                         - For single step: (batch_size, num_points, 3)
            state: State tensor. (batch_size, seq_len, state_dim) or (batch_size, state_dim).
            time_steps: Time steps tensor. (batch_size, seq_len) or (batch_size,).
            hidden_state: (For GRU) The previous hidden state. (num_layers, batch_size, hidden_dim).

        Returns:
            - For MLP: actions tensor of shape (batch_size * seq_len, action_dim).
            - For GRU: tuple of (actions, new_hidden_state).
        """
        # --- 1. Reshape inputs for sequence processing ---
        is_sequence = point_cloud.dim() == 4
        if is_sequence:
            batch_size, seq_len = point_cloud.shape[:2]
            # Reshape from (B, S, ...) to (B*S, ...) for encoders
            point_cloud = point_cloud.reshape(batch_size * seq_len, self.num_points, 3)
            if state is not None: state = state.reshape(batch_size * seq_len, -1)
            if time_steps is not None: time_steps = time_steps.reshape(batch_size * seq_len)
        else:
            batch_size = point_cloud.shape[0]
            seq_len = 1

        # --- 2. Extract features from all modalities ---
        pc_features = self.pointnet(point_cloud)
        
        features_list = []
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
            if self.fusion_method == 'add':
                pc_features = pc_features + state_features
            else: # Concat
                features_list.append(state_features)
        
        features_list.insert(0, pc_features)

        if self.time_encoder is not None:
            if time_steps is None: raise ValueError("time_steps must be provided for time encoding")
            if self.time_encoding == 'linear':
                normalized_time = time_steps.float().unsqueeze(1) / self.max_timesteps
                time_features = self.time_encoder(normalized_time)
            else:
                time_features = self.time_encoder(time_steps.long())
            features_list.append(time_features)

        # --- 3. Fuse features ---
        # Fused features have shape (B*S, policy_input_dim)
        fused_features = torch.cat(features_list, dim=1)
        
        # --- 4. Pass features to the appropriate policy head ---
        if self.policy_head_type == 'mlp':
            # MLP directly processes the flattened batch of features
            return self.policy_head(fused_features)

        elif self.policy_head_type == 'gru':
            # Reshape features for GRU: (B*S, D) -> (B, S, D)
            gru_input = fused_features.view(batch_size, seq_len, -1)
            
            # The GRU head handles both sequence and single-step inputs
            actions, new_hidden_state = self.policy_head(gru_input, hidden_state)
            return actions, new_hidden_state
        

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def create_policy(policy_type='multimodal', **kwargs):
    """
    Factory function to create different types of policies.
    Filters kwargs according to the policy constructor's expected arguments.
    """

    if policy_type == 'multimodal':
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
        mlp_hidden_dims=model_cfg.get('mlp_hidden_dims', [256, 128]),
        hidden_dims=model_cfg.get('hidden_dims', [256, 128]),
        dropout_rate=model_cfg.get('dropout_rate', 0.3),
        use_residual=model_cfg.get('use_residual', False),
        output_activation=model_cfg.get('output_activation', None),
        policy_head_type=model_cfg.get('policy_head_type', 'mlp'),
        # For multimodal policy
        state_dim=model_cfg.get('state_dim', 0),
        fusion_method=model_cfg.get('fusion_method', 'concat'),
        # for time encoding
        time_encoding=model_cfg.get('time_encoding', 'none'),
        time_embedding_dim=model_cfg.get('time_embedding_dim', 128),
        max_timesteps=model_cfg.get('max_timesteps', 64),
        # for rnn
        gru_hidden_dim=model_cfg.get('gru_hidden_dim', 0),
        gru_num_layers=model_cfg.get('gru_num_layers', 0), 
    )
    return model

# # Base policy configuration
# type: multimodal         # Options: pointcloud, multimodal
# num_points: 1000         # Number of points in each point cloud
# feature_dim: 256         # Feature dimension from PointNet
# action_dim: 3            # Dimension of action space (SE3 9D representation)
# mlp_hidden_dims: [256, 128]  # Hidden dimensions for policy MLP head
# dropout_rate: 0.2        # Dropout rate for regularization
# use_residual: false      # Whether to use residual connections in MLP
# output_activation: null  # Output activation function (null for linear output)
# policy_head_type: gru
# gru_hidden_dim: 256  # Hidden dimension for GRU
# gru_num_layers: 2  # Number of GRU layers

# # Multimodal policy specific settings
# state_dim: 3            # Same as action_dim for now (maybe change/omit)
# fusion_method: concat  # Method to fuse modalities (concat, add) 

# #time encodings
# time_encoding: positional
# max_timesteps: 64
# time_embedding_dim: 128
