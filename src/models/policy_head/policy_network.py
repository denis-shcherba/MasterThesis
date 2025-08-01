import torch
import torch.nn as nn

from models.perception.pointnet import PointNet
from models.perception.depthimageencoder import DepthImageEncoder
from models.policy_head.mlp_head import MLPHead, GRUHead, ResidualMLPHead
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
                 observation_mode='points',
                 # --- New arguments for selecting and configuring the policy head ---
                 policy_head_type: str = 'mlp', 
                 mlp_hidden_dims: Optional[list] = None,
                 gru_hidden_dim: int = 256, 
                 gru_num_layers: int = 2, 
                 use_residual: bool = False,
                 ):
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
        self.observation_mode = observation_mode
        # --- PointNet feature extractor ---
        
        if self.observation_mode == 'points':
            self.obs_encoder = PointNet(num_points=num_points, feature_dim=feature_dim, dropout_rate=dropout_rate)
        elif self.observation_mode == 'depth':
            self.obs_encoder = DepthImageEncoder(feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")
                
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
            if use_residual:
                self.policy_head = ResidualMLPHead(
                    input_dim=policy_input_dim, output_dim=action_dim,
                    hidden_dims=mlp_hidden_dims, dropout_rate=dropout_rate
                )
            else:
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
    
    def forward(self, observation: torch.Tensor, state: Optional[torch.Tensor] = None, 
                time_steps: Optional[torch.Tensor] = None, 
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy. Handles both single-step and sequence data.

        Args:
            point_cloud: Either point cloud or depth image tensor depending on observation_mode.
                - If points: shape (B, [S], N, 3)
                - If depth: shape (B, [S], 1, H, W)
            state: Optional robot state input.
            time_steps: Optional time step indices (used if time_encoding is enabled).
            hidden_state: Previous hidden state for GRU policies.

        Returns:
            - For MLP: actions of shape (B * S, action_dim)
            - For GRU: (actions, new_hidden_state)
        """
        # --- 1. Determine if input is a sequence ---
        is_sequence = observation.dim() == 3 if self.observation_mode == 'points' else observation.dim() == 4   # look into
        if is_sequence:
            batch_size, seq_len = observation.shape[:2]
            obs_input = observation.reshape(batch_size * seq_len, *observation.shape[2:])
            # if state is not None:
            #     state = state.reshape(batch_size * seq_len, -1)
            if time_steps is not None:
                time_steps = time_steps.reshape(batch_size * seq_len)
        else:
            batch_size = observation.shape[0]
            seq_len = 1
            obs_input = observation

        # --- 2. Extract features based on modality ---
        if self.observation_mode == 'points':
            obs_features = self.obs_encoder(obs_input)  # shape: (B*S, feature_dim)
        elif self.observation_mode == 'depth':
            #obs_input = self.depth_resize(obs_input)
            obs_input = obs_input.unsqueeze(1)
            obs_features = self.obs_encoder(obs_input)
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode}")

        features_list = [obs_features]

        # --- 3. Encode state ---
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
            if self.fusion_method == 'add':
                features_list[0] = features_list[0] + state_features
            else:  # concat
                features_list.append(state_features)

        # --- 4. Encode time if enabled ---
        if self.time_encoder is not None:
            if time_steps is None:
                raise ValueError("time_steps must be provided for time encoding")
            if self.time_encoding == 'linear':
                normalized_time = time_steps.float().unsqueeze(1) / self.max_timesteps
                time_features = self.time_encoder(normalized_time)
            else:
                time_features = self.time_encoder(time_steps.long())
            features_list.append(time_features)

        # --- 5. Fuse all features ---
        fused_features = torch.cat(features_list, dim=1)  # (B*S, D)

        # --- 6. Policy head ---
        if self.policy_head_type == 'mlp':
            return self.policy_head(fused_features)

        elif self.policy_head_type == 'gru':
            gru_input = fused_features.view(batch_size, seq_len, -1)
            actions, new_hidden_state = self.policy_head(gru_input, hidden_state)
            return actions, new_hidden_state

class SimplePCToPosRegressor(nn.Module):
    """
    A much simpler policy network that regresses a point cloud directly
    to a 3-element key EE-position.
    
    It uses a PointNet to extract features from the point cloud and
    then a simple linear layer for regression.
    """
    
    def __init__(self, num_points: int = 1024, feature_dim: int = 256):
        """
        Initialize the SimplifiedPolicy network.
        
        Args:
            num_points (int): Number of points expected in the input point cloud.
            feature_dim (int): Dimensionality of the features extracted by PointNet.
        """
        super(SimplePCToPosRegressor, self).__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.output_dim = 3 # Fixed output for end position (x, y, z)
        
        # PointNet feature extractor
        self.pointnet = PointNet(num_points=num_points, feature_dim=feature_dim)
        
        # Regression head: maps the global point cloud features to the 3D end position
        self.regressor_head = nn.Linear(feature_dim, self.output_dim)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the simplified policy.
        
        Args:
            point_cloud (torch.Tensor): Input point cloud tensor of shape
                                        (batch_size, num_points, 3).
        
        Returns:
            torch.Tensor: Predicted end position tensor of shape (batch_size, 3).
        """
        # 1. Extract global features from the point cloud using PointNet
        # pc_features will have shape (batch_size, feature_dim)
        pc_features = self.pointnet(point_cloud)
        
        # 2. Regress the features to the 3D end position
        # end_position will have shape (batch_size, 3)
        end_position = self.regressor_head(pc_features)
        
        return end_position



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
    elif policy_type == 'regression':
        filtered = filter_kwargs(SimplePCToPosRegressor.__init__, kwargs)
        return SimplePCToPosRegressor(**filtered)
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
        policy_type=model_cfg.get('type'),
        num_points=model_cfg.get('num_points', 1024),
        feature_dim=model_cfg.get('feature_dim', 256),
        action_dim=model_cfg.get('action_dim', 7),
        mlp_hidden_dims=model_cfg.get('mlp_hidden_dims', [256, 128]),
        hidden_dims=model_cfg.get('hidden_dims', [256, 128]),
        dropout_rate=model_cfg.get('dropout_rate', 0.3),
        use_residual=model_cfg.get('use_residual', False),
        output_activation=model_cfg.get('output_activation', None),
        policy_head_type=model_cfg.get('policy_head_type', 'mlp'),
        observation_mode=model_cfg.get('observation_mode'),
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
