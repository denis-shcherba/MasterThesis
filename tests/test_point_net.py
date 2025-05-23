import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointNet(nn.Module):
    """
    Simple PointNet implementation for point cloud feature extraction.
    Takes a point cloud of shape (batch_size, num_points, 3) and outputs
    global features of shape (batch_size, feature_dim).
    """
    
    def __init__(self, num_points=1024, feature_dim=256):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Shared MLPs for point-wise feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
        # Optional: Add a small MLP head for further processing
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, feature_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through PointNet
        Args:
            x: Point cloud tensor of shape (batch_size, num_points, 3)
        Returns:
            Global features of shape (batch_size, feature_dim)
        """
        batch_size = x.size(0)
        
        # Transpose for conv1d: (batch_size, 3, num_points)
        x = x.transpose(2, 1)
        
        # Apply shared MLPs with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling to get permutation invariant features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Optional: Apply MLP head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class PointCloudPolicy(nn.Module):
    """
    Example policy network that uses PointNet features
    """
    
    def __init__(self, num_points=1024, feature_dim=256, action_dim=6):
        super(PointCloudPolicy, self).__init__()
        
        # PointNet feature extractor
        self.pointnet = PointNet(num_points, feature_dim)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, point_cloud):
        """
        Forward pass through the policy
        Args:
            point_cloud: (batch_size, num_points, 3)
        Returns:
            actions: (batch_size, action_dim)
        """
        features = self.pointnet(point_cloud)
        actions = self.policy_head(features)
        return actions

def preprocess_point_cloud(points, target_num_points=1024):
    """
    Simple preprocessing function for point clouds
    Args:
        points: numpy array of shape (N, 3) where N is number of points
        target_num_points: desired number of points after processing
    Returns:
        processed_points: numpy array of shape (target_num_points, 3)
    """
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Normalize to unit sphere
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    # Sample or pad to target number of points
    num_points = points.shape[0]
    
    if num_points >= target_num_points:
        # Random sampling
        indices = np.random.choice(num_points, target_num_points, replace=False)
        points = points[indices]
    else:
        # Pad by repeating points
        num_repeats = target_num_points - num_points
        repeat_indices = np.random.choice(num_points, num_repeats, replace=True)
        repeated_points = points[repeat_indices]
        points = np.vstack([points, repeated_points])
    
    return points

if __name__ == "__main__":

    batch_size = 4
    num_points = 1024

    raw_points = np.load("point_cloud.npy")
    processed_points = preprocess_point_cloud(raw_points, target_num_points=1024)
    
    print(f"\nOriginal points shape: {raw_points.shape}")
    print(f"Processed points shape: {processed_points.shape}")
    
    point_tensor = torch.from_numpy(processed_points).float().unsqueeze(0)
    
    policy = PointCloudPolicy(num_points=num_points, feature_dim=256, action_dim=6)

    with torch.no_grad():
        action = policy(point_tensor)
        print(f"Single inference - Action: {action.squeeze().numpy()}")