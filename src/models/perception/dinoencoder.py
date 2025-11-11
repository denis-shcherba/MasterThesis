import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """
    MLP Adapter to map pre-extracted DINO CLS features to the Transformer's embedding dimension.
    
    Input shape: (Batch_size, Seq_len, D_dino) 
    Output shape: (Batch_size * Seq_len, D_policy)
    """
    def __init__(self, dino_feature_dim=768, feature_dim=256, dropout_prob=0.2): # Added dropout_prob
        super(FeatureAdapter, self).__init__()
        self.feature_dim = feature_dim
        self.dino_feature_dim = dino_feature_dim

        self.mlp_adapter = nn.Sequential(
            nn.Linear(self.dino_feature_dim, self.dino_feature_dim // 2),  # Hidden layer
            nn.GELU(),
            nn.Dropout(p=dropout_prob),                                  
            nn.Linear(self.dino_feature_dim // 2, self.feature_dim)       # Projection layer
        )
        print(f"Initialized trainable FeatureAdapter: {self.dino_feature_dim} -> {self.feature_dim} with dropout {dropout_prob}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Pre-calculated DINO CLS feature tensor of shape (B, S, D_dino).
        Returns:
            Adapted feature vector of shape (B*S, D_policy).
        """
        # 1. Store Batch (B) and Sequence (S) dimensions for later reshaping.
        # x.shape is (B, S, D_dino)
        batch_size, seq_len, _ = x.shape
        
        # 2. Reshape the input to (B * S, D_dino). 
        # The MLP is a standard linear layer, which only operates on the last dimension.
        # We treat all time steps and all batch items as independent samples temporarily.
        x_flat = x.view(batch_size * seq_len, self.dino_feature_dim)
        
        # 3. Pass flattened data through the MLP adapter.
        # Output shape: (B * S, D_policy)
        feature_vector_flat = self.mlp_adapter(x_flat) 
        
        # NOTE: If your Transformer head expects the output to be (B, S, D_policy), 
        # you would uncomment the line below to reshape it back.
        # feature_vector_reshaped = feature_vector_flat.view(batch_size, seq_len, self.feature_dim)
        
        # Since your target output is (batch_size * seq_len, feature_dim), we return the flattened vector.
        return feature_vector_flat
    
class FeatureAdapterCNN(nn.Module):
    """
    CNN Adapter to map pre-extracted DINO Patch features 
    to the policy's embedding dimension with dropout regularization.
    
    Input shape: (Batch_size, Seq_len, Num_Patches, D_dino) -> (B, S, 36, 768)
    Output shape: (Batch_size * Seq_len, D_policy)
    """
    def __init__(self, 
                 dino_feature_dim=768, 
                 feature_dim=256, 
                 num_patches=36,
                 dropout_prob=0.2): 
        super(FeatureAdapterCNN, self).__init__()
        self.feature_dim = feature_dim
        self.dino_feature_dim = dino_feature_dim
        self.num_patches = num_patches
        self.dropout_prob = dropout_prob

        # Calculate the grid size (e.g., 36 patches -> 6x6 grid)
        self.patch_grid_size = int(self.num_patches**0.5)
        if self.patch_grid_size * self.patch_grid_size != self.num_patches:
            raise ValueError(
                f"num_patches ({self.num_patches}) must be a perfect square."
            )

        # CNN adapter with dropout after each activation
        self.cnn_adapter = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dino_feature_dim, 
                out_channels=self.dino_feature_dim // 2, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.GELU(),
            nn.Dropout2d(self.dropout_prob),  # <- spatial dropout
            
            nn.Conv2d(
                in_channels=self.dino_feature_dim // 2, 
                out_channels=self.feature_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.GELU(),
            nn.Dropout2d(self.dropout_prob),  # <- another dropout
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        print(f"Initialized CNN FeatureAdapter with dropout={self.dropout_prob}: "
              f"({self.num_patches}, {self.dino_feature_dim}) -> {self.feature_dim}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Pre-calculated DINO patch feature tensor.
               Expected shape: (B, S, Num_Patches, D_dino)
        Returns:
            Adapted feature vector of shape (B*S, D_policy)
        """
        batch_size, seq_len, num_patches, dino_dim = x.shape
        
        if num_patches != self.num_patches or dino_dim != self.dino_feature_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (B, S, {self.num_patches}, "
                f"{self.dino_feature_dim}), but got {x.shape}"
            )
            
        # (B, S, 36, 768) -> (B*S, 36, 768)
        x_flat_batch = x.view(batch_size * seq_len, self.num_patches, self.dino_feature_dim)
        
        # (B*S, 36, 768) -> (B*S, 768, 6, 6)
        x_conv_input = x_flat_batch.permute(0, 2, 1).view(
            batch_size * seq_len, 
            self.dino_feature_dim, 
            self.patch_grid_size, 
            self.patch_grid_size
        )
        
        feature_vector_flat = self.cnn_adapter(x_conv_input)
        return feature_vector_flat