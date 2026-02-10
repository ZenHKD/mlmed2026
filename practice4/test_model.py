import torch
from model import RetinaU2NET3d

def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def format_params(num):
    """Format parameter count with units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)

if __name__ == "__main__":
    print("=" * 60)
    print("3D Retina U^2-Net Model Test")
    print("=" * 60)

    # Configuration
    batch_size = 1
    in_channels = 1
    out_channels = 1
    # D, H, W 
    d, h, w = 32, 64, 64 
    
    print("\n[1] Creating model...")
    model = RetinaU2NET3d(in_ch=in_channels, out_ch=out_channels, num_classes=1, num_anchors=1)
    
    # Check parameters
    total_params = count_parameters(model)
    print(f"    Total Parameters: {format_params(total_params)} ({total_params:,})")
    
    print("\n[2] Model Architecture Summary:")
    print("-" * 40)

    # Encoder parameters
    encoder_modules = [model.stage1, model.stage2, model.stage3, model.stage4, model.stage5, model.stage6]
    encoder_params = sum(count_parameters(m) for m in encoder_modules)
    print(f"    Encoder: {format_params(encoder_params)} ({encoder_params:,})")
    
    # Decoder parameters
    decoder_modules = [model.stage5d, model.stage4d, model.stage3d, model.stage2d, model.stage1d]
    decoder_params = sum(count_parameters(m) for m in decoder_modules)
    print(f"    Decoder: {format_params(decoder_params)} ({decoder_params:,})")
    
    reg_head_params = count_parameters(model.regression_head)
    cls_head_params = count_parameters(model.classification_head)
    print(f"    Regression Head:  {format_params(reg_head_params)} ({reg_head_params:,})")
    print(f"    Class Head:       {format_params(cls_head_params)} ({cls_head_params:,})")
    
    # Side outputs and fusion
    side_modules = [model.side1, model.side2, model.side3, model.side4, model.side5, model.side6, model.outconv]
    side_params = sum(count_parameters(m) for m in side_modules)
    print(f"    Side & Fusion:    {format_params(side_params)} ({side_params:,})")
    
    print("-" * 40)
    calculated_total = encoder_params + decoder_params + reg_head_params + cls_head_params + side_params
    print(f"    Sum of parts:     {format_params(calculated_total)} ({calculated_total:,})")

    # Input tensor
    x = torch.randn(batch_size, in_channels, d, h, w)
    print(f"\n[3] Running forward pass...")
    print(f"    Input shape: {x.shape}")
    
    masks, cls_preds, reg_preds = model(x)
    
    print("\n[4] Output shapes:")
    print(f"    Masks (fused d0): {masks[0].shape}")
    print(f"    Retina Cls [0]:   {cls_preds[0].shape}")
    print(f"    Retina Reg [0]:   {reg_preds[0].shape}")
    
    print("\n" + "=" * 60)
    print("Test finished!")
    print("=" * 60)
