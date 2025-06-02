import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointNet(nn.Module):
    """
    PointNet implementation for point cloud feature extraction.
    Takes a point cloud of shape (batch_size, num_points, 3) and outputs
    global features of shape (batch_size, feature_dim).
    
    Based on PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    by Charles R. Qi et al.
    """
    
    def __init__(self, num_points=1024, feature_dim=256, dropout_rate=0.3):
        """
        Initialize PointNet architecture.
        
        Args:
            num_points (int): Expected number of points in input point cloud
            feature_dim (int): Dimension of output global features
            dropout_rate (float): Dropout rate for regularization
        """
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Shared MLPs for point-wise feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(feature_dim)
        
        # Post-processing MLP
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, feature_dim)
        
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass through PointNet.
        
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
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        # Global max pooling to get permutation invariant features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Post-processing MLP
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def preprocess_point_cloud(points, target_num_points=1024, normalize=True, center=True):
    """
    Preprocess point cloud for PointNet input.
    
    Args:
        points: numpy array of shape (N, 3) where N is number of points
        target_num_points: desired number of points after processing
        normalize: whether to normalize point cloud to unit sphere
        center: whether to center point cloud at origin
        
    Returns:
        processed_points: numpy array of shape (target_num_points, 3)
    """
    points = points.copy()
    
    # Center the point cloud
    if center:
        centroid = np.mean(points, axis=0)
        points = points - centroid
    
    # Normalize to unit sphere
    if normalize:
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
    
    # Sample or pad to target number of points
    num_points = points.shape[0]
    
    if num_points >= target_num_points:
        # Random sampling without replacement
        indices = np.random.choice(num_points, target_num_points, replace=False)
        points = points[indices]
    else:
        # Pad by repeating points randomly
        num_repeats = target_num_points - num_points
        repeat_indices = np.random.choice(num_points, num_repeats, replace=True)
        repeated_points = points[repeat_indices]
        points = np.vstack([points, repeated_points])
    
    return points


def collate_point_clouds(point_clouds, target_num_points=1024):
    """
    Collate a batch of point clouds into a single tensor.
    
    Args:
        point_clouds: list of numpy arrays, each of shape (N_i, 3)
        target_num_points: target number of points for each point cloud
        
    Returns:
        batched_points: torch tensor of shape (batch_size, target_num_points, 3)
    """
    batch_size = len(point_clouds)
    processed_clouds = []
    
    for pc in point_clouds:
        processed_pc = preprocess_point_cloud(pc, target_num_points)
        processed_clouds.append(processed_pc)
    
    # Stack into batch tensor
    batched_points = np.stack(processed_clouds, axis=0)
    return torch.from_numpy(batched_points).float()