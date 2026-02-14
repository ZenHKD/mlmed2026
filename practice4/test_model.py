"""Test both models: 3D U-Net and 3D ResNet-18."""
import torch
from model import UNet3D, ResNet3D18

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fmt(n):
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

if __name__ == "__main__":
    print("=" * 50)
    print("Stage 1: 3D U-Net (Lung Segmentation)")
    print("=" * 50)
    unet = UNet3D(in_ch=1, out_ch=1)
    print(f"Parameters: {fmt(count_params(unet))} ({count_params(unet):,})")
    
    x = torch.randn(1, 1, 64, 256, 256)
    print(f"Input:  {x.shape}")
    y = unet(x)
    print(f"Output: {y.shape}")
    
    print()
    print("=" * 50)
    print("Stage 2: 3D ResNet-18 (Candidate Classification)")
    print("=" * 50)
    resnet = ResNet3D18(in_ch=1, num_classes=1)
    print(f"Parameters: {fmt(count_params(resnet))} ({count_params(resnet):,})")
    
    x = torch.randn(4, 1, 32, 32, 32)
    print(f"Input:  {x.shape}")
    y = resnet(x)
    print(f"Output: {y.shape}")
    
    print()
    print(f"Total: {fmt(count_params(unet) + count_params(resnet))}")