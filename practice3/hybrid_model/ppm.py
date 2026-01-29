# Pyramid Pooling Module

import torch 
import torch.nn as nn
from torch.nn import functional as F


class PPMBranch(nn.Module):
    """Single pyramid level: adaptive pool + 1x1 conv + upsample"""
    
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Upsampled features back to original H, W
        """
        _, _, H, W = x.shape
        out = self.conv(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class PyramidPoolingModule(nn.Module):
    """
    4-level Pyramid Pooling Module.
    
    Pools at 1x1, 2x2, 3x3, 6x6 scales, then concatenates
    with original features and reduces channels.
    """
    
    def __init__(self, in_channels, reduction_channels=192, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        
        self.pool_sizes = pool_sizes
        self.branches = nn.ModuleList([
            PPMBranch(in_channels, reduction_channels, size)
            for size in pool_sizes
        ])
        
        # Channel reduction after concatenation
        # in_channels + len(pool_sizes) * reduction_channels -> in_channels
        concat_channels = in_channels + len(pool_sizes) * reduction_channels
        self.reduce = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: Bottleneck features (B, C, H, W)
        Returns:
            Enhanced features with multi-scale context (B, C, H, W)
        """
        # Get pooled features from each branch
        pooled = [branch(x) for branch in self.branches]
        
        # Concatenate with original features
        concat = torch.cat([x] + pooled, dim=1)
        
        # Reduce channels
        out = self.reduce(concat)
        return out


if __name__ == "__main__":
    # Test PPM
    ppm = PyramidPoolingModule(in_channels=768, reduction_channels=192)
    x = torch.randn(2, 768, 8, 8)
    out = ppm(x)
    
    print("PPM Test:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")