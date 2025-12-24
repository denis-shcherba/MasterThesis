import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialSoftmax(nn.Module):
    """
    Converts feature maps into (x, y) coordinates (keypoints).
    Input: (B, C, H, W)
    Output: (B, C, 2) -> (x, y) coordinates in range [-1, 1]
    """
    def __init__(self, height, width, temperature=None):
        super().__init__()
        self.height = height
        self.width = width
        
        # 1. Create Meshgrid
        # indexing='ij': 
        # grid_y varies along dimension 0 (height)
        # grid_x varies along dimension 1 (width)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        
        # Register buffers (flattened)
        self.register_buffer("grid_x", grid_x.reshape((height * width)))
        self.register_buffer("grid_y", grid_y.reshape((height * width)))

        # if temperature:
        #     self.temperature = nn.Parameter(torch.ones(1) * temperature)
        # else:
        self.temperature = 1.0

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        
        # Flatten spatial dims: (B, C, H*W)
        x_flat = x.view(b, c, -1)
        
        # Apply Temperature (optional, helps sharpen heatmaps)
        if isinstance(self.temperature, nn.Parameter):
            x_flat = x_flat * self.temperature

        # Softmax
        attention = F.softmax(x_flat, dim=-1)
        
        # Calculate expected (x, y)
        # We want output to be (x, y) for grid_sample compatibility
        expected_x = torch.sum(self.grid_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.grid_y * attention, dim=-1, keepdim=True)
        
        # Concatenate: (B, C, 2)
        # Note: We do NOT flatten the last two dims here
        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        
        return keypoints

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
                 feature_dim=256,   
                 num_patches=256,   
                 num_keypoints=32, 
                 dropout_prob=0.1): # Lower dropout for small data
        super().__init__()
        
        self.patch_grid_size = int(num_patches**0.5) # e.g., 16
        self.num_keypoints = num_keypoints
        
        # 1. Compress DINO channels slightly, but keep enough for semantics
        self.compress = nn.Sequential(
            nn.Conv2d(dino_feature_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128) # Helps with training stability on small data
        )
        
        # 2. Keypoint Predictor (Heatmaps)
        # We process the compressed features to predict where the points are
        self.heatmap_predictor = nn.Conv2d(128, num_keypoints, kernel_size=3, padding=1)
        
        # 3. Output Projector
        # Input: (num_keypoints * 2 coords) + (num_keypoints * 128 features)
        # We keep the features found at the keypoints!
        self.out_projection = nn.Linear(num_keypoints * (2 + 128), feature_dim) 
        
        upsampled_size = self.patch_grid_size * 2 

        # Initialize with the correct spatial dimensions
        self.spatial_softmax = SpatialSoftmax(height=upsampled_size, width=upsampled_size)

    def forward(self, x):
        # Input x: (B, S, Num_Patches, D_dino)
        batch_size, seq_len, num_patches, dino_dim = x.shape
        x = x.view(batch_size * seq_len, num_patches, dino_dim)
        
        # Reshape to grid: (B*S, 768, 16, 16)
        x = x.permute(0, 2, 1).view(batch_size * seq_len, dino_dim, self.patch_grid_size, self.patch_grid_size)
        
        # 1. Compress features: (B*S, 128, 16, 16)
        features = self.compress(x)
        
        # 2. UPSAMPLE for precision (Bilinear) -> (B*S, 128, 32, 32)
        # doubling resolution reduces quantization error significantly
        features_up = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 3. Predict Heatmaps on upsampled grid: (B*S, 32, 32, 32)
        heatmaps = self.heatmap_predictor(features_up)
        
        # 4. Spatial Softmax -> Get (x, y) coordinates
        # shape: (B*S, num_keypoints, 2)
        keypoints_xy = self.spatial_softmax(heatmaps) 
        
        # 5. FEATURE PEEKING (Critical step)
        # We want to know WHAT is at these coordinates, not just WHERE they are.
        # Sample the 'features_up' map at the 'keypoints_xy' locations.
        # This gives us a descriptor for every keypoint.
        keypoint_features = self.sample_features_at_coords(features_up, keypoints_xy)
        
        # Flatten: 32 points * (2 coords + 128 features)
        combined = torch.cat([keypoints_xy, keypoint_features], dim=2)

        # Then flatten
        out = combined.view(batch_size * seq_len, -1)
        
        # Project to policy dim
        return self.out_projection(out)

    def sample_features_at_coords(self, feature_map, coords):
        """
        feature_map: (N, C, H, W)
        coords: (N, K, 2) normalized to [-1, 1] usually, or [0, H/W].
        SpatialSoftmax usually outputs [-1, 1]. Check your implementation!
        If your SpatialSoftmax outputs pixel coords, normalize them here.
        Assuming coords are in range [-1, 1] for grid_sample.
        """
        # grid_sample expects (N, H, W, 2)
        # We need to reshape coords to (N, 1, K, 2) to sample K points
        k = coords.shape[1]
        grid = coords.unsqueeze(1) # (N, 1, K, 2)
        
        # Sample: (N, C, 1, K)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        
        # Result: (N, K, C)
        return sampled.squeeze(2).permute(0, 2, 1)