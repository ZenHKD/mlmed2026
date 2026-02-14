"""
LUNA16 Two-Stage Models:
  Stage 1: Lightweight 3D U-Net for lung segmentation
  Stage 2: 3D ResNet-18 for nodule candidate classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Stage 1: Lightweight 3D U-Net
# ============================================================

class ConvBlock3D(nn.Module):
    """Double convolution block: Conv3d -> GroupNorm -> ReLU x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # GroupNorm works better than BatchNorm for batch_size=1
        num_groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Lightweight 3D U-Net for lung segmentation.
    
    4-level encoder-decoder with skip connections.
    Channels: 16 -> 32 -> 64 -> 128
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock3D(in_ch, 16)
        self.enc2 = ConvBlock3D(16, 32)
        self.enc3 = ConvBlock3D(32, 64)
        self.enc4 = ConvBlock3D(64, 128)  # Bottleneck
        
        self.pool = nn.MaxPool3d(2, stride=2)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock3D(128, 64)  # 64 (up) + 64 (skip)
        
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock3D(64, 32)   # 32 (up) + 32 (skip)
        
        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock3D(32, 16)   # 16 (up) + 16 (skip)
        
        # Output
        self.out_conv = nn.Conv3d(16, out_ch, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # (B, 32, D, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, D/2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 128, D/4, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 256, D/8, H/8, W/8)
        
        # Decoder with skip connections
        d3 = self.up3(e4)
        # Handle odd spatial dimensions
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)  # (B, 1, D, H, W) — logits


# ============================================================
#  Stage 2: 3D ResNet-18 Classifier
# ============================================================

class BasicBlock3D(nn.Module):
    """ResNet BasicBlock adapted for 3D."""
    expansion = 1
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        num_groups = min(8, out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D18(nn.Module):
    """Lightweight 3D ResNet-18 for nodule candidate classification.
    
    Input: (B, 1, 32, 32, 32) isotropic patch
    Output: (B, 1) nodule probability logit
    Uses narrower channels (32-64-128-256) for ~4M parameters
    """
    def __init__(self, in_ch=1, num_classes=1):
        super().__init__()
        
        # Stem (narrower: 32 channels instead of 64)
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        
        # ResNet layers with narrow channels (1 block each for speed)
        self.layer1 = self._make_layer(32, 32, blocks=1, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=1, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=1, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=1, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            num_groups = min(8, out_ch)
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
        
        layers = [BasicBlock3D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)       # (B, 32, 8, 8, 8)
        x = self.layer1(x)     # (B, 32, 8, 8, 8)
        x = self.layer2(x)     # (B, 64, 4, 4, 4)
        x = self.layer3(x)     # (B, 128, 2, 2, 2)
        x = self.layer4(x)     # (B, 256, 1, 1, 1)
        x = self.avgpool(x)    # (B, 256, 1, 1, 1)
        x = x.flatten(1)       # (B, 256)
        x = self.dropout(x)
        x = self.fc(x)         # (B, 1)
        return x
