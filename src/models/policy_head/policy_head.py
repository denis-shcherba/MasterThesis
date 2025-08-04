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


class GRUHead(nn.Module):
    """
    A recurrent policy head using a GRU to maintain a memory of past states.
    Takes a sequence of feature vectors and outputs a sequence of action predictions.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout_rate: float = 0.3, 
                 output_activation: Optional[str] = None):
        """
        Initialize the GRU-based policy head.
        
        Args:
            input_dim (int): Dimension of the input features for each time step.
            output_dim (int): Dimension of the output actions.
            hidden_dim (int): The number of features in the hidden state of the GRU.
            num_layers (int): Number of recurrent layers.
            dropout_rate (float): If non-zero, introduces a Dropout layer on the outputs of each
                                  GRU layer except the last layer.
            output_activation (str, optional): Output activation ('tanh', 'sigmoid', etc.). Defaults to None.
        """
        super(GRUHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # The core of our policy is now a GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Crucial for handling (batch, seq, feature) shaped data
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # A linear layer to map the GRU's output to the action space
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Optional output activation
        self.output_activation = self._get_activation(output_activation) if output_activation else None

        self._initialize_weights()

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get an activation function module by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid()
        }
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        return activations[activation_name]

    def _initialize_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'gru' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU head.
        
        Args:
            x: Input tensor.
               - During training (whole trajectory): (batch_size, sequence_length, input_dim)
               - During inference (single step): (batch_size, 1, input_dim)
            hidden_state: The hidden state from the previous time step.
                          Shape: (num_layers, batch_size, hidden_dim)
                          If None, it will be initialized to zeros.
            
        Returns:
            A tuple containing:
            - actions (torch.Tensor): The output actions. Shape is the same as the input's
                                      batch and sequence dimensions.
            - new_hidden_state (torch.Tensor): The new hidden state to be passed to the next step.
                                               Shape: (num_layers, batch_size, hidden_dim)
        """
        # The GRU layer returns the output for each time step and the final hidden state.
        gru_out, new_hidden_state = self.gru(x, hidden_state)
        
        # We pass the GRU's output through our final fully-connected layer.
        actions = self.fc_out(gru_out)

        if self.output_activation:
            actions = self.output_activation(actions)
            
        return actions, new_hidden_state


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
    def __init__(self, input_dim, output_dim, context_length, embed_dim=128, 
                 num_layers=4, num_heads=4, dropout=0.1, output_activation='tanh'):
        """
        Transformer-based policy head.

        Args:
            input_dim: Dim of each input vector (e.g. obs dim)
            output_dim: Dim of each output vector (e.g. action dim)
            context_length: Number of timesteps in the input sequence
            embed_dim: Dim of internal transformer embeddings
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_activation: Activation function on output ('tanh', 'sigmoid', or None)
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, context_length, embed_dim))  # learnable positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.output_activation = self._get_activation(output_activation)

    def _get_activation(self, name):
        if name is None:
            return nn.Identity()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_length, input_dim)
        Returns:
            Tensor of shape (batch_size, context_length, output_dim)
        """
        x = self.input_proj(x) + self.pos_emb  # add positional encoding
        x = self.transformer(x)
        x = self.output_proj(x)
        return self.output_activation(x)
