import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- SegNetVGG16 Model Definition ---
class SegNetVGG16(nn.Module):
    """
    SegNet model with a VGG-16 backbone.
    The encoder is replaced by VGG-16's convolutional layers.
    The decoder structure is maintained as per the original SegNet,
    using pooling indices from the VGG-16 encoder.
    This version outputs raw logits, suitable for nn.BCEWithLogitsLoss.
    """
    def __init__(self, num_classes=1, pretrained_vgg=True):
        """
        Initialize the SegNetVGG16 model.

        Args:
            num_classes (int): Number of output classes for segmentation.
                               Defaults to 1 (for binary segmentation).
            pretrained_vgg (bool): If True, loads VGG-16 weights pre-trained on ImageNet.
                                   Defaults to True.
        """
        super(SegNetVGG16, self).__init__()

        # Load VGG16 with batch normalization
        if pretrained_vgg:
            print("Loading VGG16_BN with pretrained weights...")
            # Try to load weights using the modern API
            try:
                vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            except AttributeError:
                # Fallback for older torchvision versions that use `pretrained=True`
                print("Using older torchvision `pretrained=True` for VGG16_BN.")
                vgg16_bn = models.vgg16_bn(pretrained=True)
        else:
            print("Initializing VGG16_BN without pretrained weights...")
            vgg16_bn = models.vgg16_bn(weights=None)

        vgg_features = vgg16_bn.features

        # Encoder part using VGG16_BN features
        # VGG16-BN architecture:
        # Block 1: layers 0-5 (2 conv layers, 64 filters), MaxPool at layer 6
        # Block 2: layers 7-12 (2 conv layers, 128 filters), MaxPool at layer 13
        # Block 3: layers 14-22 (3 conv layers, 256 filters), MaxPool at layer 23
        # Block 4: layers 24-32 (3 conv layers, 512 filters), MaxPool at layer 33
        # Block 5: layers 34-42 (3 conv layers, 512 filters), MaxPool at layer 43

        # Encoder Stage 1
        self.enc1_conv = nn.Sequential(*list(vgg_features.children())[0:6])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage 2
        self.enc2_conv = nn.Sequential(*list(vgg_features.children())[7:13])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage 3
        self.enc3_conv = nn.Sequential(*list(vgg_features.children())[14:23])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage 4
        self.enc4_conv = nn.Sequential(*list(vgg_features.children())[24:33])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage 5
        self.enc5_conv = nn.Sequential(*list(vgg_features.children())[34:43])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder blocks
        self.dec5 = self._decoder_block(512, 512)
        self.dec4 = self._decoder_block(512, 256)
        self.dec3 = self._decoder_block(256, 128)
        self.dec2 = self._decoder_block(128, 64)

        # Final decoder block
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1) # Final convolution to num_classes
        )

        # MaxUnpool layer (common for all decoder stages)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def _decoder_block(self, in_channels, out_channels):
        """
        Helper function to create a decoder block.
        Uses ConvTranspose2d for feature transformation, not upsampling here.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the SegNetVGG16 model.
        Input image size is expected to be 512x512 pixels.
        Outputs raw logits.
        """
        # Encoder Path
        x_enc1 = self.enc1_conv(x); size1 = x_enc1.size()
        x_pool1, indices1 = self.pool1(x_enc1)

        x_enc2 = self.enc2_conv(x_pool1); size2 = x_enc2.size()
        x_pool2, indices2 = self.pool2(x_enc2)

        x_enc3 = self.enc3_conv(x_pool2); size3 = x_enc3.size()
        x_pool3, indices3 = self.pool3(x_enc3)

        x_enc4 = self.enc4_conv(x_pool3); size4 = x_enc4.size()
        x_pool4, indices4 = self.pool4(x_enc4)

        x_enc5 = self.enc5_conv(x_pool4); size5 = x_enc5.size()
        x_pool5, indices5 = self.pool5(x_enc5)

        # Decoder Path
        x_dec5_unpooled = self.unpool(x_pool5, indices5, output_size=size5)
        x_dec5 = self.dec5(x_dec5_unpooled)

        x_dec4_unpooled = self.unpool(x_dec5, indices4, output_size=size4)
        x_dec4 = self.dec4(x_dec4_unpooled)

        x_dec3_unpooled = self.unpool(x_dec4, indices3, output_size=size3)
        x_dec3 = self.dec3(x_dec3_unpooled)

        x_dec2_unpooled = self.unpool(x_dec3, indices2, output_size=size2)
        x_dec2 = self.dec2(x_dec2_unpooled)

        x_dec1_unpooled = self.unpool(x_dec2, indices1, output_size=size1)
        out = self.dec1(x_dec1_unpooled) # Outputting logits

        return out