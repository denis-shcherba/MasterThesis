import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import numpy as np

class DINOCLSEncoder(nn.Module):
    """
    DINOv2 ViT-based encoder for depth images, using the CLS token and an MLP adapter.
    
    The DINO backbone is frozen, and only the MLP adapter is trained.
    Input: (B, 1, H, W) -> will be replicated to (B, 3, H, W) for DINO
    Output: (B, feature_dim)
    """
    def __init__(self, feature_dim=256, dino_model_name='facebook/dinov2-base'):
        super(DINOCLSEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.dino_model_name = dino_model_name

        # --- 1. Load Pre-trained DINOv2 Model and Freeze ---
        try:
            # We only need the model, the AutoImageProcessor will be used in the forward pass
            self.dino_model = AutoModel.from_pretrained(self.dino_model_name)
        except Exception as e:
            raise ImportError(f"Failed to load DINOv2 model. Ensure you have 'transformers' installed. Error: {e}")

        # Freeze the entire DINO backbone (crucial for sim2real benefit and training stability)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        print(f"Loaded and froze DINOv2 backbone: {self.dino_model_name}")

        # Determine the input feature dimension from the DINO model config
        dino_feature_dim = self.dino_model.config.hidden_size 
        
        # --- 2. Define the MLP Adapter (Trainable) ---
        # This acts as the replacement for the ResNet's final FC layer.
        # A simple 2-layer MLP often works well for adaptation.
        self.mlp_adapter = nn.Sequential(
            nn.Linear(dino_feature_dim, dino_feature_dim // 2),  # Hidden layer
            nn.GELU(),
            nn.Linear(dino_feature_dim // 2, feature_dim)       # Projection layer
        )
        print(f"MLP Adapter: {dino_feature_dim} -> {dino_feature_dim // 2} -> {feature_dim}")


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Depth image tensor of shape (B, 1, H, W). Expects values to be [0, 1] or [0, 255].
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        # --- 1. Prepare Input (Single-channel Depth to 3-channel) ---
        # Replicate the single depth channel three times to fit the RGB-trained DINO model.
        # DINO expects the input to be in the format it was trained on (e.g., normalized, etc.),
        # but the AutoModel handles most of the internal normalization. The crucial step
        # here is getting the channel count right.
        
        # Ensure the input is float and replicate the channel dimension (1 -> 3)
        # Shape: (B, 1, H, W) -> (B, 3, H, W)
        x_3ch = x.repeat(1, 3, 1, 1).float() 

        # --- 2. DINO Feature Extraction ---
        # Pass the 3-channel input through the frozen DINO model
        with torch.no_grad():
            outputs = self.dino_model(x_3ch)
            
        # Extract the CLS token: The first token in the sequence (index 0) is the CLS token
        # last_hidden_state shape is (B, num_tokens + 1, hidden_size)
        # We select index 0 on the token dimension (dim=1)
        cls_token = outputs.last_hidden_state[:, 0, :] # Shape: (B, dino_feature_dim)

        # --- 3. MLP Adapter Projection (Trainable) ---
        # The CLS token is passed to the small, trainable adapter
        feature_vector = self.mlp_adapter(cls_token) # Shape: (B, feature_dim)
        
        return feature_vector

# --- Example Usage (Conceptual) ---
# Assuming you want a 512-dimensional feature vector
# encoder = DINOCLSEncoder(feature_dim=512)
# dummy_input = torch.randn(4, 1, 224, 224) # 4 images, 1 channel, 224x224
# output_features = encoder(dummy_input)
# print(f"Output shape: {output_features.shape}") # Should be torch.Size([4, 512])