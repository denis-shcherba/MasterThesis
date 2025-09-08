import torch
import torch.nn as nn

from models.perception.pointnet import PointNet
from models.perception.depthimageencoder import DepthImageEncoder
from models.policy_head.policy_head import MLPHead, GRUHead, ResidualMLPHead, TransformerHead
import math
import inspect
from typing import Optional, Tuple

def prepare_depth_input(observation: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Ensures depth input is in shape (B*S, 1, H, W)
    Returns: obs_input, batch_size, seq_len
    """
    if observation.dim() == 4:  # (B, S, H, W)
        batch_size, seq_len = observation.shape[:2]
        obs_input = observation.view(batch_size * seq_len, 1, *observation.shape[2:])
    elif observation.dim() == 3:  # (B, H, W)
        batch_size = observation.shape[0]
        seq_len = 1
        obs_input = observation.unsqueeze(1)  # (B, 1, H, W)
    else:
        raise ValueError(f"Unexpected depth observation shape: {observation.shape}")
    return obs_input, batch_size, seq_len


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
        
        # (1, max_timesteps, embedding_dim) for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, time_steps):
        """
        Args:
            time_steps: Tensor of shape (batch_size,) or (batch_size, seq_len)
                        with integer timestep indices (e.g., [0, 1, 2, ...])
        Returns:
            Positional encodings of shape (batch_size, embedding_dim) or
            (batch_size, seq_len, embedding_dim)
        """
        if time_steps.dim() == 1:
            return self.pe[0, time_steps]  # (batch_size, embedding_dim)
        elif time_steps.dim() == 2:
            return self.pe[0, time_steps]  # (batch_size, seq_len, embedding_dim)
        else:
            raise ValueError("time_steps must be of shape (batch,) or (batch, seq_len)")

class WayPlusTimingsPolicy(nn.Module):
    """
    Multi-head policy:
      - Transformer head on depth sequence -> predicts timings
      - N waypoint regressors on the *first depth image* -> predicts key waypoints
    """

    def __init__(self,
                 feature_dim=256,
                 state_dim=0,
                 action_dim=6,
                 dropout_rate=0.3,
                 fusion_method='concat',
                 context_length=5,
                 embed_dim=256,
                 num_layers=4,
                 num_heads=4,
                 num_waypoints=2,   # <--- NEW
                 waypoint_dim=3     # <--- xyz
                 ):
        super().__init__()

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim

        # Depth encoder with regularization
        self.obs_encoder = DepthImageEncoder(
            feature_dim=feature_dim, 
            freeze_layers=True,      # Freeze layer3 and layer4
            dropout_rate=0.1         # Lower dropout for encoder
        )

        # Optional state encoder
        self.state_encoder = nn.Linear(state_dim, feature_dim) if state_dim > 0 else None
        policy_input_dim = feature_dim + (feature_dim if state_dim > 0 and fusion_method == 'concat' else 0)

        # Transformer head for sequence prediction
        self.transformer_head = TransformerHead(
            input_dim=policy_input_dim,
            output_dim=action_dim,
            context_length=context_length,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout_rate,
            output_activation=None,
        )

        # MLP regressors for waypoints - reduce dropout back to 0.3
        self.waypoint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 32),  # Keep smaller size
                nn.ReLU(),
                nn.Dropout(0.2),             # Reduce from 0.4 to 0.3
                nn.Linear(32, waypoint_dim)
            ) for _ in range(num_waypoints)
        ])

    def forward(self, depth_sequence: torch.Tensor,
                state: Optional[torch.Tensor] = None, first_obs: torch.Tensor = None) -> dict:
        """
        Args:
            depth_sequence: (B, T, 1, H, W)
            state: optional robot state (B, T, state_dim)

        Returns:
            dict with:
              - 'timings': (B, action_dim)
              - 'waypoints': list of (B, waypoint_dim)
        """

        B, T = depth_sequence.shape[:2]

        # Flatten time for encoder
        depth_in = depth_sequence.view(B * T, 1, *depth_sequence.shape[2:])
        features = self.obs_encoder(depth_in)              # (B*T, D)
        features = features.view(B, T, -1)                 # (B, T, D)

        # State fusion
        if state is not None and self.state_encoder is not None:
            state_f = self.state_encoder(state.view(B * T, -1)).view(B, T, -1)
            features = torch.cat([features, state_f], dim=-1)

        # Transformer forward
        timing_pred = self.transformer_head(features)[:, -1, :]  # (B, action_dim)

        # Waypoints from *first depth image only* TODO
        #first_frame = depth_sequence[:, 0, :, :].unsqueeze(1)     # (B, 1, H, W)
        first_features = self.obs_encoder(first_obs.unsqueeze(1))           # (B, D)
        waypoint_preds = [head(first_features) for head in self.waypoint_heads]

        return {
            "timings": timing_pred,
            "waypoints": waypoint_preds
        }



class MultiModalPolicy(nn.Module):
    """
    Policy network that handles multiple input modalities and supports both
    stateless (MLP) and stateful (GRU/Transformer) policy heads.
    """

    def __init__(self, num_points=1024, feature_dim=256,
                 state_dim=0, action_dim=6,
                 dropout_rate=0.3, fusion_method='concat',
                 time_encoding='none', time_embedding_dim=256, max_timesteps=64,
                 observation_mode='points',
                 policy_head_type: str = 'mlp',
                 mlp_hidden_dims: Optional[list] = None,
                 gru_hidden_dim: int = 256,
                 gru_num_layers: int = 2,
                 use_residual: bool = False,
                 # for Transformer
                 context_length: int = None,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 output_activation: str = "tanh",
                 ):
        super(MultiModalPolicy, self).__init__()
        
        # Store configuration
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fusion_method = fusion_method
        self.time_encoding = time_encoding
        self.max_timesteps = max_timesteps
        self.policy_head_type = policy_head_type
        self.observation_mode = observation_mode



        # Observation encoder
        if self.observation_mode == 'points':
            self.obs_encoder = PointNet(num_points=num_points, feature_dim=feature_dim, dropout_rate=dropout_rate)
        elif self.observation_mode == 'depth':
            self.obs_encoder = DepthImageEncoder(feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")

        # State encoder
        self.state_encoder = nn.Linear(state_dim, feature_dim) if state_dim > 0 else None

        # Determine the input dimension for the policy head
        policy_input_dim = feature_dim
        if state_dim > 0:
            self.state_encoder = nn.Linear(state_dim, feature_dim)
            if fusion_method == 'concat':
                policy_input_dim += feature_dim
            elif fusion_method == 'concat_project':
                # This fusion projects back down to feature_dim, so policy_input_dim doesn't change
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(feature_dim + feature_dim, 512), # obs_dim + state_dim
                    nn.ReLU(),
                    nn.Linear(512, feature_dim), # output matches obs_feature_dim
                )
            # For 'add' fusion, policy_input_dim also doesn't change

        # IMPORTANT: Only add time encoding for non-transformer heads
        # Transformer handles positional encoding internally
        if policy_head_type != 'transformer' and time_encoding != 'none':
            self.time_encoder = self._create_time_encoder(time_encoding, time_embedding_dim, max_timesteps)
            policy_input_dim += time_embedding_dim
        else:
            self.time_encoder = None

        # Create the selected policy head
        if policy_head_type == 'mlp':
            if mlp_hidden_dims is None: 
                mlp_hidden_dims = [256, 128]
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

        elif policy_head_type == 'transformer':
            if context_length is None:
                context_length = max_timesteps
            
            # For transformer, we don't add external time encoding
            # The transformer handles positional encoding internally
            
            self.policy_head = TransformerHead(
                input_dim=policy_input_dim,  # No time encoding added here
                output_dim=action_dim,
                context_length=context_length,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout_rate,
                output_activation=None,
            )
        else:
            raise ValueError(f"Unknown policy_head_type: {policy_head_type}")

    def _create_time_encoder(self, encoding_type, embedding_dim, max_steps):
        if encoding_type == 'none': 
            return None
        if encoding_type == 'positional': 
            return PositionalEncoding(embedding_dim, max_steps)
        if encoding_type == 'embedding': 
            return nn.Embedding(max_steps, embedding_dim)
        if encoding_type == 'linear': 
            return nn.Linear(1, embedding_dim)
        raise ValueError(f"Unknown time_encoding method: {encoding_type}")

    def forward(self, observation: torch.Tensor, state: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy. Handles both single-step and sequence data.
        
        Args:
            observation: Point cloud or depth image tensor depending on observation_mode.
                - If points: shape (B, [S], N, 3)  
                - If depth: shape (B, [S], 1, H, W)
            state: Optional robot state input.
            time_steps: Optional time step indices (used if time_encoding is enabled).
            hidden_state: Previous hidden state for GRU policies.
            
        Returns:
            - For MLP: actions of shape (B * S, action_dim)
            - For GRU: (actions, new_hidden_state)
            - For Transformer: actions of shape (B, action_dim) for last timestep
        """

        if self.observation_mode == 'depth':
            obs_input, batch_size, seq_len = prepare_depth_input(observation)
            obs_features = self.obs_encoder(obs_input)  # (B*S, D)

            if time_steps is not None:
                time_steps = time_steps.view(batch_size * seq_len)

        elif self.observation_mode == 'points':
            obs_features = self.obs_encoder(obs_input)  # (B*S, feature_dim)
        elif self.observation_mode == 'depth':
            obs_input = obs_input.unsqueeze(1)
            obs_features = self.obs_encoder(obs_input)
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode}")

        features_list = [obs_features]

        # Encode state
        if state is not None and self.state_encoder is not None:

            state = state.reshape(batch_size * seq_len, -1)
            state_features = self.state_encoder(state)

            if self.fusion_method == 'add':
                features_list[0] = features_list[0] + state_features

            elif self.fusion_method == 'concat_project':
                # Concatenate and project via fusion MLP
                fusion_input = torch.cat([obs_features, state_features], dim=1)
                fused = self.fusion_mlp(fusion_input)  # (B*S, D)
                features_list[0] = fused  # replace obs_features with fused
            else:  # vanilla concat
                features_list.append(state_features)


        # Encode time if enabled (ONLY for non-transformer heads)
        if self.time_encoder is not None:
            if time_steps is None:
                raise ValueError("time_steps must be provided for time encoding")
            if self.time_encoding == 'linear':
                normalized_time = time_steps.float().unsqueeze(1) / self.max_timesteps
                time_features = self.time_encoder(normalized_time)
            else:
                time_features = self.time_encoder(time_steps.long())
            features_list.append(time_features)

        # Fuse all features
        fused_features = torch.cat(features_list, dim=1)  # (B*S, D)

        # Policy head forward pass
        if self.policy_head_type == 'mlp':
            return self.policy_head(fused_features)

        elif self.policy_head_type == 'gru':
            gru_input = fused_features.view(batch_size, seq_len, -1)
            actions, new_hidden_state = self.policy_head(gru_input, hidden_state)
            return actions, new_hidden_state

        elif self.policy_head_type == 'transformer':
            # Reshape for transformer: (B, T, D)
            seq_input = fused_features.view(batch_size, seq_len, -1)
            
            # Your transformer handles positional encoding internally
            action_seq = self.policy_head(seq_input)  # (B, T, action_dim)
            
            # Return last action (most common use case)
            # You could also return the full sequence if needed
            return action_seq[:, -1, :]  # (B, action_dim)

        else:
            raise ValueError(f"Unknown policy_head_type: {self.policy_head_type}")
        
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
    elif policy_type == 'wayplustiming':
        filtered = filter_kwargs(WayPlusTimingsPolicy.__init__, kwargs)
        return WayPlusTimingsPolicy(**filtered)
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
        # for transformer
        context_length=model_cfg.get('context_length', None),
        embed_dim=model_cfg.get('embed_dim', 256),
        num_layers=model_cfg.get('num_layers', 4),
        num_heads=model_cfg.get('num_heads', 4),
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
