import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.models import resnet18, ResNet18_Weights, resnet101, ResNet101_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        out = torch.mul(feat, atten)
        return out

class ContextPath(nn.Module):
    def __init__(self, backbone, in_channels_16, in_channels_32, conv_head8_in_channels):
        super(ContextPath, self).__init__()
        self.backbone = backbone
        
        self.in_channels_16 = in_channels_16
        self.in_channels_32 = in_channels_32
        
        # ARM modules with correct input channels
        self.arm16 = AttentionRefinementModule(self.in_channels_16, 128)
        self.arm32 = AttentionRefinementModule(self.in_channels_32, 128)
        
        # Additional conv layer to convert feat32_gp to 128 channels
        self.conv_gp = ConvBlock(self.in_channels_32, 128, kernel_size=1, padding=0)
        
        # Output conv layers
        self.conv_head8 = ConvBlock(conv_head8_in_channels, 128)
        self.conv_head16 = ConvBlock(128, 128)
        self.conv_head32 = ConvBlock(128, 128)
        
    def forward(self, x):
        # Extract features using ResNet
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # low-level features
        feat4 = self.backbone.layer1(x)  # 1/4
        # mid-level features
        feat8 = self.backbone.layer2(feat4)  # 1/8
        # high-level features
        feat16 = self.backbone.layer3(feat8)  # 1/16
        feat32 = self.backbone.layer4(feat16)  # 1/32
        
        # attention refinement
        feat16_arm = self.arm16(feat16)
        feat32_arm = self.arm32(feat32)
        
        # global average pooling
        feat32_gp = F.avg_pool2d(feat32, feat32.size()[2:])
        # Convert channels from 512 to 128 to match feat32_arm
        feat32_gp = self.conv_gp(feat32_gp)
        feat32_gp = F.interpolate(feat32_gp, size=feat32.size()[2:], mode='bilinear', align_corners=True)
        
        # fusion of attention features
        feat32_sum = feat32_arm + feat32_gp
        feat32_up = F.interpolate(feat32_sum, size=feat16.size()[2:], mode='bilinear', align_corners=True)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.size()[2:], mode='bilinear', align_corners=True)
        
        # output features
        feat8_out = self.conv_head8(feat8)
        feat16_out = self.conv_head16(feat16_sum)
        feat32_out = self.conv_head32(feat32_sum)
        
        return feat8_out, feat16_out, feat32_out

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, spatial_feat, context_feat):
        feat = torch.cat([spatial_feat, context_feat], dim=1)
        feat = self.conv1(feat)
        
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        
        feat_atten = torch.mul(feat, atten)
        out = feat + feat_atten
        
        return out

class BiSeNet(nn.Module):
    def __init__(self, num_classes=1, backbone_type='resnet18'):
        super(BiSeNet, self).__init__()
        self.backbone_type = backbone_type.lower()
        
        if self.backbone_type == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            in_channels_16 = 256
            in_channels_32 = 512
            conv_head8_in_channels = 128
        elif self.backbone_type == 'resnet101':
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            in_channels_16 = 1024
            in_channels_32 = 2048
            conv_head8_in_channels = 512
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Supported: 'resnet18', 'resnet101'")
        
        # Spatial Path
        self.spatial_path = SpatialPath()
        
        # Context Path with dynamic channel sizes
        self.context_path = ContextPath(backbone, in_channels_16, in_channels_32, conv_head8_in_channels)
        
        # Feature Fusion Module
        self.ffm = FeatureFusionModule(256 + 128, 256)
        
        # Output layer
        self.output_layer = nn.Sequential(
            ConvBlock(256, 256),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Auxiliary output layers for deep supervision
        self.output_aux1 = nn.Sequential(
            ConvBlock(128, 128),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        self.output_aux2 = nn.Sequential(
            ConvBlock(128, 128),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        # Spatial Path
        spatial_feat = self.spatial_path(x)
        
        # Context Path
        context_feats = self.context_path(x)
        feat8, feat16, feat32 = context_feats
        context_feat = feat8
        
        # Feature Fusion
        fusion_feat = self.ffm(spatial_feat, context_feat)
        
        # Output
        output = self.output_layer(fusion_feat)
        output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
        
        # Auxiliary outputs
        output_aux1 = self.output_aux1(feat16)
        output_aux1 = F.interpolate(output_aux1, size=size, mode='bilinear', align_corners=True)
        
        output_aux2 = self.output_aux2(feat32)
        output_aux2 = F.interpolate(output_aux2, size=size, mode='bilinear', align_corners=True)
        
        return output, output_aux1, output_aux2

class WaterSegLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.15, gamma=0.15):
        super(WaterSegLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        
    def forward(self, outputs, target):
        output, output_aux1, output_aux2 = outputs
        
        target_float = target.float()
        
        main_bce = self.bce_loss(output, target_float)
        main_dice = self.dice_loss(torch.sigmoid(output), target_float)
        main_loss = main_bce + main_dice
        
        aux1_bce = self.bce_loss(output_aux1, target_float)
        aux1_dice = self.dice_loss(torch.sigmoid(output_aux1), target_float)
        aux1_loss = aux1_bce + aux1_dice
        
        aux2_bce = self.bce_loss(output_aux2, target_float)
        aux2_dice = self.dice_loss(torch.sigmoid(output_aux2), target_float)
        aux2_loss = aux2_bce + aux2_dice
        
        total_loss = self.alpha * main_loss + self.beta * aux1_loss + self.gamma * aux2_loss
        
        return total_loss
