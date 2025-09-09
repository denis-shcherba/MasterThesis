import torch
import torch.nn as nn
import torchvision.models as models


class DepthImageEncoder(nn.Module):
    """
    ResNet-based encoder for depth images with regularization. ~11M params
    Input: (B, 1, H, W)
    Output: (B, feature_dim)
    """
    def __init__(self, feature_dim=256, pretrained=False, freeze_layers=False):
        super(DepthImageEncoder, self).__init__()
        self.feature_dim = feature_dim

        # Start with a ResNet18 (or use resnet34, etc.)
        base_model = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept 1 channel (depth image)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = base_model.conv1.weight.data.sum(dim=1, keepdim=True) / 3.0  # average RGB weights

        # Keep rest of the layers
        self.encoder = nn.Sequential(
            self.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,  # Output shape: (B, 512, 1, 1)
        )

        # Freeze later layers to reduce overfitting
        if freeze_layers:
            # Freeze layer3 and layer4 (the deepest, most specific layers)
            for param in self.encoder[6].parameters():  # layer3
                param.requires_grad = False
            for param in self.encoder[7].parameters():  # layer4
                param.requires_grad = False
            print("Frozen ResNet layer3 and layer4")

        # Final projection layer
        self.fc = nn.Linear(512, feature_dim)

        # # Final projection layer with dropout
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(512, feature_dim)
        # )

    def forward(self, x):
        """
        Args:
            x: Depth image tensor of shape (B, 1, H, W)
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        x = self.encoder(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.fc(x)  # (B, feature_dim)
        return x



class SimpleDepthEncoder(nn.Module):
    """
    Simple CNN encoder for depth images - much smaller than ResNet18. ~500k params
    Input: (B, 1, H, W)
    Output: (B, feature_dim)
    """
    def __init__(self, feature_dim=256, dropout_rate=0.2, input_size=(224, 224)):
        super(SimpleDepthEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        # Simple CNN backbone
        self.encoder = nn.Sequential(
            # First conv block: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # H/4, W/4
            
            # Second conv block: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # H/8, W/8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # H/16, W/16
            
            # Third conv block: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # H/32, W/32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth conv block: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # H/64, W/64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        )
        
        # Final projection with dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Depth image tensor of shape (B, 1, H, W)
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        x = self.encoder(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)  # (B, feature_dim)
        return x