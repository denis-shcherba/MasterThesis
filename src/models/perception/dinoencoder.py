import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialSoftmax(nn.Module):
    """
    Converts feature maps into (x, y) coordinates (keypoints).
    Input: (B, C, H, W)
    Output: (B, C * 2) -> A set of x,y coords for each channel.
    """
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        
        # Create a meshgrid of pixel coordinates [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        # Register them as buffers so they move to GPU automatically
        self.register_buffer("pos_x", pos_x.reshape((height * width)))
        self.register_buffer("pos_y", pos_y.reshape((height * width)))

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        
        # Flatten spatial dims: (B, C, H*W)
        x_flat = x.view(b, c, -1)
        
        # Apply Softmax over the spatial dimension to get attention maps
        # This makes the map sum to 1, treating it like a probability distribution
        attention = F.softmax(x_flat, dim=-1)
        
        # Calculate expected value (weighted average) of coordinates
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)
        
        # Concatenate x and y to get (B, C, 2) then flatten to (B, C*2)
        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        return keypoints.reshape(b, -1)

class FeatureAdapter(nn.Module):
    def __init__(self, dino_feature_dim=768, feature_dim=256, dropout_prob=0.5):
        super().__init__()
        # SINGLE linear layer - no hidden layer
        self.adapter = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(dino_feature_dim, feature_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(batch_size * seq_len, -1)
        return self.adapter(x_flat)
    
    
class FeatureAdapterSpatial(nn.Module):
    def __init__(self, 
                 dino_feature_dim=768, 
                 feature_dim=256,   # Matches your old output dim
                 num_patches=64,    # Matches your old 6x6 grid
                 num_keypoints=32,  # Internal bottleneck (can be 32 or 64)
                 dropout_prob=0.3): 
        super().__init__()
        
        self.patch_grid_size = int(num_patches**0.5)
        # 1. Compress Channels (768 -> 32)
        # We reduce channels first to force the model to pick only the 32 most 
        # important "types" of features to track spatially.
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dino_feature_dim, 256, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout_prob), # Dropout on features, not pixels
            nn.Conv2d(256, num_keypoints, kernel_size=1) 
        )
        # 2. Extract Coordinates (32 maps -> 32 x,y pairs)
        self.spatial_softmax = SpatialSoftmax(self.patch_grid_size, self.patch_grid_size)
        
        # 3. Project to Policy Dim (64 -> 256)
        # We linearly project the 64 coords back up to 256 to match your 
        # original policy input size.
        self.out_projection = nn.Linear(num_keypoints * 2, feature_dim) 
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        print(f"Spatial Adapter Initialized: Output shape (Batch, {feature_dim})")

    def forward(self, x):
        # Input x: (B, S, Num_Patches, D_dino)
        batch_size, seq_len, num_patches, dino_dim = x.shape
        
        # Flatten time: (B*S, Num_Patches, D_dino)
        x = x.view(batch_size * seq_len, num_patches, dino_dim)
        
        # Reshape to grid: (B*S, 768, 6, 6)
        x = x.permute(0, 2, 1).view(batch_size * seq_len, dino_dim, self.patch_grid_size, self.patch_grid_size)
        
        # 1. Convolution: (B*S, 32, 6, 6)
        x = self.conv1x1(x)
        
        # 2. Spatial Softmax: (B*S, 64) 
        # This contains the raw (x,y) coords of the 32 keypoints
        x = x * self.temperature
        x = self.spatial_softmax(x)
        
        # 3. Linear Projection: (B*S, 256)
        x = self.out_projection(x)
        
        return x