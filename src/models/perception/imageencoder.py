import torch.nn as nn
from torchvision import models

class FlexibleImageEncoder(nn.Module):
    """
    ResNet-based encoder for both RGB and Depth images.
    Input: (B, input_channels, H, W)
    """
    def __init__(self, input_channels=1, feature_dim=256, pretrained=False, dropout_rate=0.2):
        super(FlexibleImageEncoder, self).__init__()
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        base_model = models.resnet18(weights=weights)

        # LOGIC FOR INPUT CHANNELS
        if input_channels == 3:
            # If RGB, just use the original layer
            self.conv1 = base_model.conv1 
        else:
            # If Depth/Grayscale (1 channel), modify the layer and average weights
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Collapse 3 channels into 1 by averaging
                self.conv1.weight.data = base_model.conv1.weight.data.sum(dim=1, keepdim=True) / 3.0

        # Construct the encoder using our selected conv1
        self.encoder = nn.Sequential(
            self.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool, 
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x