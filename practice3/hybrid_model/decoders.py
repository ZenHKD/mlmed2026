# Decoder & Cross-Attention

import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    """
    Efficient channel-wise cross-attention between two feature maps.
    
    Uses global pooling + channel attention instead of full spatial attention
    to be memory-efficient. O(C^2) instead of O((H*W)^2).
    
    Query comes from one decoder, Key/Value from the other decoder.
    """
    
    def __init__(self, channels, num_heads=4, reduction=4):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 16)
        
        # Channel attention via SE-like mechanism
        self.query_pool = nn.AdaptiveAvgPool2d(1)
        self.kv_pool = nn.AdaptiveAvgPool2d(1)
        
        # Cross-attention MLP
        self.cross_attn = nn.Sequential(
            nn.Linear(channels * 2, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
        # Spatial refinement with depthwise conv
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, query_feat, kv_feat):
        """
        Args:
            query_feat: Features from current decoder (B, C, H, W)
            kv_feat: Features from other decoder (B, C, H, W)
        Returns:
            Cross-attended features (B, C, H, W)
        """
        B, C, H, W = query_feat.shape
        
        # Global pooling for channel descriptors
        q_desc = self.query_pool(query_feat).view(B, C)  # (B, C)
        kv_desc = self.kv_pool(kv_feat).view(B, C)       # (B, C)
        
        # Cross-attention: combine descriptors to get channel weights
        combined = torch.cat([q_desc, kv_desc], dim=1)   # (B, 2C)
        channel_weights = self.cross_attn(combined).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Apply channel attention to kv features
        attended = kv_feat * channel_weights
        
        # Spatial refinement
        out = self.spatial_refine(attended)
        
        return out


class UpConvBlock(nn.Module):
    """
    Upsampling block with skip connection.
    
    Upsamples features 2x, concatenates with skip features, 
    then applies conv blocks.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsample 2x
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convs after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: Features to upsample (B, C_in, H, W)
            skip: Skip connection features (B, C_skip, 2H, 2W)
        Returns:
            Upsampled + fused features (B, C_out, 2H, 2W)
        """
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    """
    Complete decoder block with UpConv + CrossAttention + Residual.
    
    Flow: UpConv -> CrossAttn -> Add (residual) -> output
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, num_heads=4):
        super().__init__()
        
        self.upconv = UpConvBlock(in_channels, skip_channels, out_channels)
        self.cross_attn = CrossAttention(out_channels, num_heads)
        self.norm = nn.LayerNorm(out_channels)
        
        # For residual: match channels if needed
        self.residual_proj = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, skip, other_feat):
        """
        Args:
            x: Input features (B, C_in, H, W)
            skip: Skip connection from encoder (B, C_skip, 2H, 2W)
            other_feat: Features from the other decoder for cross-attention (B, C_out, 2H, 2W)
        Returns:
            Enhanced features with cross-attention (B, C_out, 2H, 2W)
        """
        # Upsample and fuse with skip
        feat = self.upconv(x, skip)
        
        # Cross-attention with other decoder
        cross_feat = self.cross_attn(feat, other_feat)
        
        # Residual addition
        out = self.norm((feat + cross_feat).permute(0,2,3,1)).permute(0,3,1,2)
        
        return out


class DualDecoder(nn.Module):
    """
    Dual decoder with cross-attention at each scale.
    
    Both lung and infection decoders share the same structure,
    but exchange information via cross-attention.
    """
    
    def __init__(self, encoder_channels=[96, 192, 384, 768]):
        super().__init__()
        
        # Decoder channels (reverse of encoder)
        dec_channels = [384, 192, 96, 48]
        
        # Lung decoder blocks
        self.lung_blocks = nn.ModuleList([
            UpConvBlock(encoder_channels[3], encoder_channels[2], dec_channels[0]),  # 768+384 -> 384
            UpConvBlock(dec_channels[0], encoder_channels[1], dec_channels[1]),       # 384+192 -> 192
            UpConvBlock(dec_channels[1], encoder_channels[0], dec_channels[2]),       # 192+96 -> 96
            UpConvBlock(dec_channels[2], 0, dec_channels[3]),                          # 96 -> 48 (no skip)
        ])
        
        # Infection decoder blocks
        self.inf_blocks = nn.ModuleList([
            UpConvBlock(encoder_channels[3], encoder_channels[2], dec_channels[0]),
            UpConvBlock(dec_channels[0], encoder_channels[1], dec_channels[1]),
            UpConvBlock(dec_channels[1], encoder_channels[0], dec_channels[2]),
            UpConvBlock(dec_channels[2], 0, dec_channels[3]),
        ])
        
        # Cross-attention at each scale
        self.lung_cross_attn = nn.ModuleList([
            CrossAttention(dec_channels[0], num_heads=8),
            CrossAttention(dec_channels[1], num_heads=4),
            CrossAttention(dec_channels[2], num_heads=2),
            CrossAttention(dec_channels[3], num_heads=1),
        ])
        
        self.inf_cross_attn = nn.ModuleList([
            CrossAttention(dec_channels[0], num_heads=8),
            CrossAttention(dec_channels[1], num_heads=4),
            CrossAttention(dec_channels[2], num_heads=2),
            CrossAttention(dec_channels[3], num_heads=1),
        ])
        
        self.out_channels = dec_channels[3]
    
    def forward(self, bottleneck, encoder_features):
        """
        Args:
            bottleneck: PPM output (B, 768, H/32, W/32)
            encoder_features: List of encoder stage outputs [s1, s2, s3, s4]
        Returns:
            lung_feat: Lung decoder output (B, 48, H/2, W/2)
            inf_feat: Infection decoder output (B, 48, H/2, W/2)
        """
        # Reverse encoder features for skip connections [s4, s3, s2, s1]
        skips = encoder_features[::-1]
        
        # Initialize both decoders with bottleneck
        lung_x = bottleneck
        inf_x = bottleneck
        
        # Process each scale
        for i in range(4):
            # Get skip connection (empty for last stage)
            if i < 3:
                skip = skips[i + 1]
            else:
                # Create dummy skip for last stage (just zeros)
                _, _, H, W = lung_x.shape
                skip = torch.zeros(lung_x.size(0), 0, H * 2, W * 2, device=lung_x.device)
            
            # UpConv for both decoders
            lung_feat = self.lung_blocks[i](lung_x, skip)
            inf_feat = self.inf_blocks[i](inf_x, skip)
            
            # Cross-attention: lung attends to infection, infection attends to lung
            lung_cross = self.lung_cross_attn[i](lung_feat, inf_feat)
            inf_cross = self.inf_cross_attn[i](inf_feat, lung_feat)
            
            # Residual addition
            lung_x = lung_feat + lung_cross
            inf_x = inf_feat + inf_cross
        
        return lung_x, inf_x


if __name__ == "__main__":
    # Test decoder
    decoder = DualDecoder()
    
    # Simulate encoder outputs
    bottleneck = torch.randn(2, 768, 8, 8)
    encoder_features = [
        torch.randn(2, 96, 64, 64),
        torch.randn(2, 192, 32, 32),
        torch.randn(2, 384, 16, 16),
        torch.randn(2, 768, 8, 8),
    ]
    
    lung, inf = decoder(bottleneck, encoder_features)
    
    print("Dual Decoder Test:")
    print(f"  Bottleneck: {bottleneck.shape}")
    print(f"  Lung output: {lung.shape}")
    print(f"  Infection output: {inf.shape}")
