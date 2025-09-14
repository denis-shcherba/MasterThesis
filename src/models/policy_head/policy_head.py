import torch
import torch.nn as nn
from typing import Optional, Tuple


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for policy networks.
    Takes feature vectors and outputs action predictions.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.3, 
                 activation='relu', output_activation=None):
        """
        Initialize MLP head.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output actions
            hidden_dims (list): List of hidden layer dimensions. If None, uses [256, 128]
            dropout_rate (float): Dropout rate for regularization
            activation (str): Activation function ('relu', 'tanh', 'elu')
            output_activation (str): Output activation function (None, 'tanh', 'sigmoid')
        """
        super(MLPHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation
        if output_activation is not None:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through MLP head.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


### --- OMIT Maybe ---

class ResidualMLPHead(nn.Module):
    """
    MLP head with residual connections for deeper networks.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.3,
                 activation='relu', output_activation=None):
        """
        Initialize Residual MLP head.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output actions
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
            activation (str): Activation function
            output_activation (str): Output activation function
        """
        super(ResidualMLPHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection if needed
        self.input_proj = None
        if input_dim != hidden_dims[0]:
            self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(
                hidden_dims[i], hidden_dims[i+1], 
                dropout_rate=dropout_rate, activation=activation
            )
            self.residual_blocks.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Output activation
        self.output_activation = None
        if output_activation is not None:
            self.output_activation = self._get_activation(output_activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through Residual MLP head.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input projection
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for deeper MLP networks.
    """
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, activation='relu'):
        """
        Initialize residual block.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            dropout_rate (float): Dropout rate
            activation (str): Activation function
        """
        super(ResidualBlock, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main path
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Skip connection
        self.skip_connection = None
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        
        # Activation and dropout
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """Forward pass through residual block."""
        residual = x
        
        # Main path
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        
        # Add residual and apply activation
        out = out + residual
        out = self.activation(out)
        
        return out
    
class TransformerHead(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, 
                 prediction_length, # <-- NEW: The length 'm' of the action chunk
                 embed_dim=128,
                 num_layers=4, num_heads=4, dropout=0.1, output_activation='tanh'):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length # <-- NEW
        self.output_dim = output_dim
        
        # Action embedding for the input sequence is still useful
        self.action_embed = nn.Linear(output_dim, embed_dim)
        
        # Project observation features into the embedding dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional embedding for the input sequence
        self.pos_emb = nn.Parameter(torch.randn(1, context_length, embed_dim))

        # Standard Transformer Encoder (NO MASK)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # Important!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- KEY CHANGE: The Prediction Head ---
        # This MLP takes the final encoded representation and predicts the entire future chunk.
        self.prediction_head = nn.Linear(embed_dim, prediction_length * output_dim)
        
        self.output_activation = self._get_activation(output_activation)
        
    def _get_activation(self, name):
        if name is None: return nn.Identity()
        if name == 'tanh': return nn.Tanh()
        if name == 'sigmoid': return nn.Sigmoid()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, obs_sequence, prev_actions_sequence):
        """
        Args:
            obs_sequence: Tensor of observations, shape (B, context_length, input_dim)
            prev_actions_sequence: Tensor of previous actions, shape (B, context_length, output_dim)
        Returns:
            Tensor of PREDICTED FUTURE actions, shape (B, prediction_length, output_dim)
        """
        # 1. Embed inputs and add positional encoding
        obs_embed = self.input_proj(obs_sequence)
        action_embed = self.action_embed(prev_actions_sequence)
        x = obs_embed + action_embed + self.pos_emb
        
        # 2. Encode the entire input sequence (NO MASK)
        # The output 'encoded_seq' has shape (B, context_length, embed_dim)
        encoded_seq = self.transformer_encoder(x)
        
        # 3. Take the final hidden state as the context summary
        # This vector at the last time step has seen all previous inputs.
        context_summary = encoded_seq[:, -1, :] # Shape: (B, embed_dim)
        
        # 4. Use the prediction head to generate the future chunk
        predicted_chunk_flat = self.prediction_head(context_summary) # Shape: (B, m * action_dim)
        
        # 5. Reshape to the desired output format
        predicted_chunk = predicted_chunk_flat.view(
            -1, self.prediction_length, self.output_dim
        ) # Shape: (B, m, action_dim)
        
        return self.output_activation(predicted_chunk)
