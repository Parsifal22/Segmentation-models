
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


class SegNetWithMobileNetV3(nn.Module):
    def __init__(self, num_classes=1, mode='large', pretrained=True):
        super(SegNetWithMobileNetV3, self).__init__()
        self.mode = mode
        
        # Load pre-trained MobileNetV3 as backbone
        if self.mode == 'large':
            self.backbone = mobilenet_v3_large(pretrained=pretrained)
            # Define feature channels for each level based on MobileNetV3 architecture
            self.feat1_channels = 24    # Early features
            self.feat2_channels = 40    # Low-level features 
            self.feat3_channels = 112   # Mid-level features
            self.feat4_channels = 960   # High-level features
        else:  # small
            self.backbone = mobilenet_v3_small(pretrained=pretrained)
            # Define feature channels for each level based on MobileNetV3 architecture
            self.feat1_channels = 16    # Early features
            self.feat2_channels = 24    # Low-level features
            self.feat3_channels = 48    # Mid-level features
            self.feat4_channels = 576   # High-level features
        
        # Remove the classifier part of the model
        self.backbone = self.backbone.features
        
        # Additional bottleneck layer for high-level features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.feat4_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.dec4 = self._decoder_block(512 + self.feat3_channels, self.feat2_channels)
        self.dec3 = self._decoder_block(self.feat2_channels, self.feat1_channels)
        self.dec2 = self._decoder_block(self.feat1_channels, 64)
        self.dec1 = self._decoder_block(64, 64)
        
        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        input_size = x.size()
        features = []
        
        # Extract features from the backbone at different stages
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Store feature maps at specific stages
            if self.mode == 'large':
                if i == 3:  # First feature level - early layers
                    feat1 = x
                elif i == 6:  # Second feature level
                    feat2 = x
                elif i == 12:  # Third feature level
                    feat3 = x
            else:  # small
                if i == 1:  # First feature level - early layers
                    feat1 = x
                elif i == 3:  # Second feature level
                    feat2 = x
                elif i == 8:  # Third feature level
                    feat3 = x
        
        # Last layer output is our high-level features
        feat4 = x
        
        # Process highest level features
        x = self.bottleneck(feat4)
        
        # Decoder with skip connections and upsampling
        # Level 4 to 3
        x = F.interpolate(x, size=feat3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, feat3], dim=1)
        x = self.dec4(x)
        
        # Level 3 to 2
        x = F.interpolate(x, size=feat2.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec3(x)
        
        # Level 2 to 1
        x = F.interpolate(x, size=feat1.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec2(x)
        
        # Level 1 to original size
        x = F.interpolate(x, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        x = self.dec1(x)
        
        # Final convolution layer
        x = self.final(x)
        
        return x
