import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TNet(nn.Module):
    """
    Transformation Network (T-Net) from original PointNet paper.
    Learns transformation matrices to canonicalize point clouds.
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        
        return x

class PointNetEncoder(nn.Module):
    """
    Faithful PointNet implementation following the original paper.
    Includes both input transform and feature transform networks.
    """
    def __init__(self, feature_dim=1024, use_input_transform=True, use_feature_transform=True):
        super(PointNetEncoder, self).__init__()
        
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform
        
        # Input Transform Network (3x3 transformation)
        if self.use_input_transform:
            self.input_transform = TNet(k=3)
        
        # First shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Feature Transform Network (64x64 transformation)
        if self.use_feature_transform:
            self.feature_transform = TNet(k=64)
        
        # Second shared MLP
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: Point cloud of shape (batch_size, num_points, 3)
        Returns:
            global_features: Global features of shape (batch_size, feature_dim)
            trans_input: Input transformation matrix (batch_size, 3, 3) or None
            trans_feature: Feature transformation matrix (batch_size, 64, 64) or None
        """
        batch_size, num_points, _ = x.size()
        
        # Transpose for conv1d operations
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)
        
        trans_input = None
        trans_feature = None
        
        # Apply input transformation
        if self.use_input_transform:
            trans_input = self.input_transform(x)  # (batch_size, 3, 3)
            x = x.transpose(2, 1)  # (batch_size, num_points, 3)
            x = torch.bmm(x, trans_input)  # Apply transformation
            x = x.transpose(2, 1)  # Back to (batch_size, 3, num_points)
        
        # First shared MLP
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, num_points)
        
        # Apply feature transformation
        if self.use_feature_transform:
            trans_feature = self.feature_transform(x)  # (batch_size, 64, 64)
            x = x.transpose(2, 1)  # (batch_size, num_points, 64)
            x = torch.bmm(x, trans_feature)  # Apply transformation
            x = x.transpose(2, 1)  # Back to (batch_size, 64, num_points)
        
        # Second shared MLP
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, num_points)
        x = self.bn3(self.conv3(x))  # (batch_size, feature_dim, num_points)
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, feature_dim, 1)
        x = x.view(batch_size, -1)  # (batch_size, feature_dim)
        
        return x, trans_input, trans_feature

def feature_transform_regularizer(trans):
    """
    Regularization loss for feature transformation matrix.
    Encourages the transformation to be close to orthogonal.
    
    Args:
        trans: Feature transformation matrix (batch_size, K, K)
    Returns:
        loss: Regularization loss
    """
    d = trans.size(1)
    batch_size = trans.size(0)
    
    I = torch.eye(d, device=trans.device, dtype=trans.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    
    return loss

class PointNetClassifier(nn.Module):
    """
    Complete PointNet for classification tasks
    """
    def __init__(self, num_classes, feature_dim=1024, dropout=0.3):
        super(PointNetClassifier, self).__init__()
        
        self.encoder = PointNetEncoder(feature_dim=feature_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        global_features, trans_input, trans_feature = self.encoder(x)
        logits = self.classifier(global_features)
        return logits, trans_input, trans_feature

class PointNetPolicy(nn.Module):
    """
    PointNet-based policy network with proper transformations
    """
    def __init__(self, action_dim=6, feature_dim=1024, dropout=0.3):
        super(PointNetPolicy, self).__init__()
        
        self.encoder = PointNetEncoder(feature_dim=feature_dim)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        global_features, trans_input, trans_feature = self.encoder(x)
        actions = self.policy_head(global_features)
        return actions, trans_input, trans_feature

# Training function example showing how to use the regularization loss
def train_step(model, point_clouds, targets, optimizer, criterion, reg_weight=0.001):
    """
    Example training step showing proper loss computation with regularization
    """
    optimizer.zero_grad()
    
    # Forward pass
    outputs, trans_input, trans_feature = model(point_clouds)
    
    # Main loss (classification or regression)
    main_loss = criterion(outputs, targets)
    
    # Regularization loss for feature transformation
    reg_loss = 0
    if trans_feature is not None:
        reg_loss = feature_transform_regularizer(trans_feature)
    
    # Total loss
    total_loss = main_loss + reg_weight * reg_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), main_loss.item(), reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0

# Example usage
if __name__ == "__main__":
    # Create example data
    batch_size = 8
    num_points = 1024
    num_classes = 10
    action_dim = 6
    
    point_clouds = torch.randn(batch_size, num_points, 3)
    
    print("=== Testing Faithful PointNet Implementation ===")
    
    # Test encoder
    encoder = PointNetEncoder(feature_dim=1024)
    features, trans_input, trans_feature = encoder(point_clouds)
    
    print(f"Input shape: {point_clouds.shape}")
    print(f"Global features shape: {features.shape}")
    if trans_input is not None:
        print(f"Input transformation shape: {trans_input.shape}")
    if trans_feature is not None:
        print(f"Feature transformation shape: {trans_feature.shape}")
    
    # Test classifier
    classifier = PointNetClassifier(num_classes=num_classes)
    logits, trans_input, trans_feature = classifier(point_clouds)
    print(f"Classification logits shape: {logits.shape}")
    
    # Test policy
    policy = PointNetPolicy(action_dim=action_dim)
    actions, trans_input, trans_feature = policy(point_clouds)
    print(f"Policy actions shape: {actions.shape}")
    
    # Test regularization loss
    if trans_feature is not None:
        reg_loss = feature_transform_regularizer(trans_feature)
        print(f"Feature transform regularization loss: {reg_loss.item():.6f}")
    
    # Test without transformations (simplified version)
    encoder_simple = PointNetEncoder(feature_dim=1024, 
                                   use_input_transform=False, 
                                   use_feature_transform=False)
    features_simple, _, _ = encoder_simple(point_clouds)
    print(f"Simplified encoder features shape: {features_simple.shape}")