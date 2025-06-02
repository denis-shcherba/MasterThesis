import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for policy network.
    Takes features from perception model and outputs actions.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 6,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        final_activation: Optional[str] = None
    ):
        super(MLPHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout (not on the last hidden layer)
            if i < len(hidden_dims) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Final activation if specified
        if final_activation is not None:
            layers.append(self._get_activation(final_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP head.
        
        Args:
            x: Input features tensor of shape (batch_size, input_dim)
            
        Returns:
            Output actions tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class ResidualMLPHead(nn.Module):
    """
    MLP head with residual connections for deeper networks.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 6,
        num_layers: int = 3,
        activation: str = 'relu',
        dropout_rate: float = 0.1
    ):
        super(ResidualMLPHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.residual_blocks.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input features tensor of shape (batch_size, input_dim)
            
        Returns:
            Output actions tensor of shape (batch_size, output_dim)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = self.activation(x + residual)  # Residual connection
        
        # Output layer
        x = self.output_layer(x)
        
        return x