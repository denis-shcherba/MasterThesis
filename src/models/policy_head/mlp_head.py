import torch
import torch.nn as nn
import torch.nn.functional as F


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