import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """
    MLP Adapter to map pre-extracted DINO CLS features to the Transformer's embedding dimension.
    
    Input shape: (Batch_size, Seq_len, D_dino) 
    Output shape: (Batch_size * Seq_len, D_policy)
    """
    def __init__(self, dino_feature_dim=768, feature_dim=256):
        super(FeatureAdapter, self).__init__()
        self.feature_dim = feature_dim
        self.dino_feature_dim = dino_feature_dim

        self.mlp_adapter = nn.Sequential(
            nn.Linear(self.dino_feature_dim, self.dino_feature_dim // 2),  # Hidden layer
            nn.GELU(),
            nn.Linear(self.dino_feature_dim // 2, self.feature_dim)       # Projection layer
        )
        print(f"Initialized trainable FeatureAdapter: {self.dino_feature_dim} -> {self.feature_dim}")

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