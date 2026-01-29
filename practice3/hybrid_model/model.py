# Multi-Task Model

import torch
import torch.nn as nn
from torch.nn import functional as F

from .encoder import SwinEncoder
from .ppm import PyramidPoolingModule
from .decoders import DualDecoder


class SegmentationHead(nn.Module):
    """Final convolution + upsample for mask output."""
    
    def __init__(self, in_channels, out_channels=1, scale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )
        self.scale_factor = scale_factor
    
    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)


class ClassificationHead(nn.Module):
    """
    Conv + GAP + FC for 3-class classification.
    
    Supports two modes:
    - Phase 1 (use_infection=False): Uses only lung mask (in_channels=1)
    - Phase 2 (use_infection=True): Uses both lung and infection masks (in_channels=2)
    """
    
    def __init__(self, num_classes=3):
        super().__init__()
        # Phase 1: lung only (1 channel)
        self.conv_lung = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Phase 2: combined (2 channels)
        self.conv_combined = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, lung_mask, inf_mask, use_infection=True):
        """
        Args:
            lung_mask: (B, 1, H, W)
            inf_mask: (B, 1, H, W)
            use_infection: If True, use both masks (Phase 2). If False, use only lung mask (Phase 1).
        Returns:
            logits: (B, num_classes)
        """
        if use_infection:
            x = torch.cat([lung_mask, inf_mask], dim=1)  # (B, 2, H, W)
            x = self.conv_combined(x)  # (B, 64, 1, 1)
        else:
            x = self.conv_lung(lung_mask)  # (B, 64, 1, 1)
        
        x = x.flatten(1)  # (B, 64)
        x = self.fc(x)    # (B, num_classes)
        return x


class MultiTaskSwinPPM(nn.Module):
    """
    Complete Multi-Task Swin-PPM Model.
    
    Architecture:
        - Swin-Tiny Encoder (pure PyTorch, no pretrained weights)
        - Pyramid Pooling Module
        - Dual Decoder with Cross-Attention
        - Segmentation Heads (Lung & Infection)
        - Classification Head
    
    Outputs:
        - lung_mask: (B, 1, H, W)
        - infection_mask: (B, 1, H, W)
        - class_logits: (B, 3)
    """
    
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        
        # Encoder (pure PyTorch, no pretrained)
        self.encoder = SwinEncoder(in_channels=in_channels)
        encoder_channels = self.encoder.out_channels  # [96, 192, 384, 768]
        
        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(
            in_channels=encoder_channels[-1],  # 768
            reduction_channels=192
        )
        
        # Dual Decoder with Cross-Attention
        self.decoder = DualDecoder(encoder_channels=encoder_channels)
        
        # Segmentation Heads
        decoder_out_channels = self.decoder.out_channels  # 48
        self.lung_head = SegmentationHead(decoder_out_channels, out_channels=1, scale_factor=2)
        self.infection_head = SegmentationHead(decoder_out_channels, out_channels=1, scale_factor=2)
        
        # Classification Head
        self.classification_head = ClassificationHead(num_classes=num_classes)
    
    def forward(self, x, use_infection=True):
        """
        Args:
            x: Input CXR image (B, 1, H, W)
            use_infection: If True (Phase 2), use both masks for classification.
                          If False (Phase 1), use only lung mask for classification.
        
        Returns:
            lung_mask: Lung segmentation mask (B, 1, H, W)
            infection_mask: Infection segmentation mask (B, 1, H, W)
            class_logits: Classification logits (B, 3) [Normal, Non-COVID, COVID]
        """
        # Encoder
        encoder_features = self.encoder(x)  # [s1, s2, s3, s4]
        
        # PPM on bottleneck
        bottleneck = self.ppm(encoder_features[-1])  # (B, 768, H/32, W/32)
        
        # Dual Decoder
        lung_feat, inf_feat = self.decoder(bottleneck, encoder_features)
        
        # Segmentation outputs
        lung_mask = self.lung_head(lung_feat)
        infection_mask = self.infection_head(inf_feat)
        
        # Classification from masks
        class_logits = self.classification_head(lung_mask, infection_mask, use_infection=use_infection)
        
        return lung_mask, infection_mask, class_logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_encoder_params(self):
        """Get encoder parameters for differential learning rate."""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder + heads parameters."""
        return list(self.ppm.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.lung_head.parameters()) + \
               list(self.infection_head.parameters()) + \
               list(self.classification_head.parameters())


if __name__ == "__main__":
    # Test complete model
    model = MultiTaskSwinPPM(in_channels=1)
    x = torch.randn(2, 1, 256, 256)
    
    # Test Phase 1 mode (lung only for classification)
    lung, inf, cls = model(x, use_infection=False)
    print("MultiTaskSwinPPM Test (Phase 1 mode):")
    print(f"  Input: {x.shape}")
    print(f"  Lung mask: {lung.shape}")
    print(f"  Infection mask: {inf.shape}")
    print(f"  Classification: {cls.shape}")
    
    # Test Phase 2 mode (both masks for classification)
    lung, inf, cls = model(x, use_infection=True)
    print("\nMultiTaskSwinPPM Test (Phase 2 mode):")
    print(f"  Classification: {cls.shape}")
    print(f"  Total parameters: {model.count_parameters():,}")