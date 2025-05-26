import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation
from torchvision import models
import timm

# Helper classes from xception_model.py
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# DeepLabV3+ with VGG16 backbone (from model_vgg.py)
class DeepLabV3Plus_VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3Plus_VGG, self).__init__()
        
        # Load pretrained VGG16 as backbone
        vgg16 = models.vgg16(pretrained=True)
        self.backbone_features = vgg16.features
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        ])
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Concatenation and 1x1 conv
        self.concat_projection = nn.Sequential(
            nn.Conv2d(256 * 5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Low-level features from VGG16
        self.low_level_features = nn.Sequential(*list(self.backbone_features)[:10])  # First 10 layers for low-level features
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 304 = 256 + 48
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Freeze batch norm layers
        self._freeze_batchnorm()
        
    def _freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        
    def forward(self, x):
        # Input size: B x 3 x 512 x 512
        
        # Backbone
        x_backbone = self.backbone_features(x)  # B x 512 x 32 x 32
        
        # Low-level features
        low_level_feat = self.low_level_features(x)  # B x 128 x 128 x 128
        low_level_feat = self.low_level_projection(low_level_feat)  # B x 48 x 128 x 128
        
        # ASPP
        aspp_outputs = [branch(x_backbone) for branch in self.aspp]
        
        # Global average pooling
        global_feat = self.global_avg_pool(x_backbone)
        global_feat = nn.functional.interpolate(global_feat, size=x_backbone.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate ASPP outputs and global feature
        aspp_outputs.append(global_feat)
        aspp_concat = torch.cat(aspp_outputs, dim=1)
        aspp_output = self.concat_projection(aspp_concat)  # B x 256 x 32 x 32
        
        # Upsample ASPP output to match low-level features size
        aspp_output = nn.functional.interpolate(aspp_output, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate with low-level features
        decoder_input = torch.cat([aspp_output, low_level_feat], dim=1)  # B x 304 x 128 x 128
        
        # Decoder
        x = self.decoder(decoder_input)  # B x num_classes x 128 x 128
        
        # Upsample to original size
        x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        
        return x

# DeepLabV3+ with timm Xception backbone (from xception_model.py)
class DeepLabV3Plus_Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3Plus_Xception, self).__init__()

        # Load pre-trained Xception model from timm
        self.backbone = timm.create_model('xception', pretrained=True, features_only=True)

        # Get feature dimensions
        # The Xception model from timm returns 5 feature levels with increasing depth
        # We use the second level for low-level features and the last level for ASPP input
        dummy_input = torch.zeros(1, 3, 520, 520) # Use a slightly larger size for dummy input
        features = self.backbone(dummy_input)

        # Feature dimensions - for proper channel dimensioning
        self.low_level_channels = features[1].shape[1]  # Block 2 features
        self.high_level_channels = features[-1].shape[1]  # Last block features

        # ASPP module
        self.aspp = ASPP(self.high_level_channels, 256, atrous_rates=[12, 24, 36])

        # Low-level features processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        # Fix batch normalization layers
        self._fix_batchnorm_layers()

    def _fix_batchnorm_layers(self):
        """Convert BatchNorm layers to use running statistics (eval mode) even during training"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
                m.eval()

    def forward(self, x):
        # Ensure input size is handled
        input_shape = x.shape[-2:]

        # Extract features from backbone
        features = self.backbone(x)

        # Get low-level features (second block) and high-level features (last block)
        low_level_feat = features[1]
        high_level_feat = features[-1]

        # Apply ASPP to high-level features
        x = self.aspp(high_level_feat)

        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample ASPP features to match low-level features size
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate features
        x = torch.cat([x, low_level_feat], dim=1)

        # Apply decoder
        x = self.decoder(x)

        # Final upsampling to match input size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

# Main DeepLabV3+ class with construct method
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.model = None
        self.num_classes = num_classes

    def construct(self, model_type='resnet101', backbone='resnet101'):
        if model_type == 'resnet':
            if backbone == 'resnet101':
                self.model = segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
            elif backbone == 'resnet50':
                self.model = segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            else:
                raise ValueError("Backbone must be either 'resnet50' or 'resnet101' for resnet model_type")
            # Replace classifier for binary segmentation (water/non-water)
            self.model.classifier = segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
            self._fix_batchnorm_layers(self.model)
        elif model_type == 'vgg16':
            self.model = DeepLabV3Plus_VGG(num_classes=self.num_classes)
        elif model_type == 'xception':
            self.model = DeepLabV3Plus_Xception(num_classes=self.num_classes)
        else:
            raise ValueError("model_type must be one of 'resnet', 'vgg16', or 'xception'")

    def _fix_batchnorm_layers(self, model):
        """Convert BatchNorm layers to use running statistics (eval mode) even during training"""
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.eval()

    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Model has not been constructed. Call construct() first.")
        
        if isinstance(self.model, (DeepLabV3Plus_VGG, DeepLabV3Plus_Xception)):
            return self.model(x)
        else: # Handle torchvision models
            output = self.model(x)
            return output['out']
