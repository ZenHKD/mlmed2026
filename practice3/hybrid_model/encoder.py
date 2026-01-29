# Swin Transformer Encoder

import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: original height and width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding: 4x4 conv with stride 4.
    Converts image to patch tokens.
    """
    
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, H/4, W/4, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, embed_dim)
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.
    Uses Flash Attention (via scaled_dot_product_attention) when available.
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop  # Store as float for SDPA
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size * window_size
            mask: (num_windows, N, N) or None
        Returns:
            x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B_, num_heads, N, head_dim)
        
        # Get relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        
        # Combine relative position bias with mask
        if mask is not None:
            nW = mask.shape[0]
            # Expand bias for batch: (1, num_heads, N, N) + reshape for windows
            attn_bias = relative_position_bias.unsqueeze(0)  # (1, num_heads, N, N)
            # Reshape mask to match: (B_//nW, nW, 1, N, N)
            mask_expanded = mask.unsqueeze(0).unsqueeze(2)  # (1, nW, 1, N, N)
            # Manually add bias per-window, reshape attn_bias
            attn_bias = attn_bias.expand(B_ // nW, -1, -1, -1).unsqueeze(1)  # (B_//nW, 1, heads, N, N)
            attn_bias = attn_bias.expand(-1, nW, -1, -1, -1)  # (B_//nW, nW, heads, N, N)
            attn_bias = attn_bias + mask_expanded  # broadcast add mask
            attn_bias = attn_bias.reshape(B_, self.num_heads, N, N)
        else:
            attn_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)  # (B_, num_heads, N, N)
        
        # Use scaled_dot_product_attention (Flash Attention when available)
        # Note: SDPA expects attn_mask to be additive (0 = attend, -inf = mask)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale
        )
        
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block: Window Attention + Shifted Window Attention.
    Caches attention mask for efficiency.
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # Cached attention mask (created on first forward pass)
        self.register_buffer("attn_mask", None, persistent=False)
        self._mask_h = None
        self._mask_w = None
    
    def _create_mask(self, H, W, device):
        """Create and cache attention mask for shifted window attention."""
        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def _get_mask(self, H, W, device):
        """Get cached mask or create new one if dimensions changed."""
        if self.shift_size == 0:
            return None
        
        # Check if need to create/recreate the mask
        if self.attn_mask is None or self._mask_h != H or self._mask_w != W:
            self.attn_mask = self._create_mask(H, W, device)
            self._mask_h = H
            self._mask_w = W
        
        return self.attn_mask
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, ws*ws, C)
        
        # Get cached attention mask
        attn_mask = self._get_mask(H, W, x.device)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, ws*ws, C)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging: Reduce spatial resolution by 2x, increase channels by 2x.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H/2, W/2, 2*C)
        """
        B, H, W, C = x.shape
        
        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


class SwinStage(nn.Module):
    """
    A Swin Transformer stage with multiple blocks.
    """
    
    def __init__(self, dim, depth, num_heads, window_size=7, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 downsample=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        
        # Downsample
        self.downsample = PatchMerging(dim) if downsample else None
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H/2, W/2, 2*C) if downsample else (B, H, W, C)
            x_out: (B, H, W, C) feature before downsampling (for skip connection)
        """
        for blk in self.blocks:
            x = blk(x)
        
        x_out = x  # Save for skip connection
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, x_out


class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder - Pure PyTorch Implementation.
    
    Architecture (Swin-Tiny):
        - Patch Embedding: 4x4, stride 4
        - Stage 1: 2 blocks, 96 channels
        - Stage 2: 2 blocks, 192 channels
        - Stage 3: 6 blocks, 384 channels
        - Stage 4: 2 blocks, 768 channels
    
    For 256x256 input:
        - Stage 1 output: (B, 96, 64, 64)
        - Stage 2 output: (B, 192, 32, 32)
        - Stage 3 output: (B, 384, 16, 16)
        - Stage 4 output: (B, 768, 8, 8)
    """
    
    def __init__(self, in_channels=1, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=8, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim)
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=(i < self.num_stages - 1)  # No downsample in last stage
            )
            self.stages.append(stage)
        
        # Output channels for each stage
        self.out_channels = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: List of (B, C, H, W) feature maps from each stage
        """
        # Patch embedding: (B, C, H, W) -> (B, H/4, W/4, embed_dim)
        x = self.patch_embed(x)
        
        features = []
        for stage in self.stages:
            x, x_out = stage(x)
            # Convert from (B, H, W, C) to (B, C, H, W) for output
            features.append(x_out.permute(0, 3, 1, 2).contiguous())
        
        return features


if __name__ == "__main__":
    # Test encoder
    print("Testing SwinEncoder (Pure PyTorch)...")
    encoder = SwinEncoder(in_channels=1)
    x = torch.randn(2, 1, 256, 256)
    features = encoder(x)
    
    print("Swin Encoder Test:")
    for i, f in enumerate(features):
        print(f"  Stage {i+1}: {f.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print("    Encoder test passed!")
