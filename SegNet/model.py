import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        
        # Encoder blocks
        self.enc1 = self._encoder_block(3, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        self.enc5 = self._encoder_block(512, 512)
        
        # Decoder blocks
        self.dec5 = self._decoder_block(512, 512)
        self.dec4 = self._decoder_block(512, 256)
        self.dec3 = self._decoder_block(256, 128)
        self.dec2 = self._decoder_block(128, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
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
        # Encoder
        x1 = self.enc1(x)
        x1_size = x1.size()
        x1_pool, indices1 = self.pool(x1)
        
        x2 = self.enc2(x1_pool)
        x2_size = x2.size()
        x2_pool, indices2 = self.pool(x2)
        
        x3 = self.enc3(x2_pool)
        x3_size = x3.size()
        x3_pool, indices3 = self.pool(x3)
        
        x4 = self.enc4(x3_pool)
        x4_size = x4.size()
        x4_pool, indices4 = self.pool(x4)
        
        x5 = self.enc5(x4_pool)
        x5_size = x5.size()
        x5_pool, indices5 = self.pool(x5)
        
        # Decoder
        x = self.unpool(x5_pool, indices5, output_size=x5_size)
        x = self.dec5(x)
        
        x = self.unpool(x, indices4, output_size=x4_size)
        x = self.dec4(x)
        
        x = self.unpool(x, indices3, output_size=x3_size)
        x = self.dec3(x)
        
        x = self.unpool(x, indices2, output_size=x2_size)
        x = self.dec2(x)
        
        x = self.unpool(x, indices1, output_size=x1_size)
        x = self.dec1(x)
        
        return x
