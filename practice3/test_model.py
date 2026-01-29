"""
Test script for MultiTaskSwinPPM model.

Verifies:
1. Input/Output shapes
2. Parameter counts per module
3. Memory usage
4. Forward pass timing
"""

import torch
import time
from hybrid_model import MultiTaskSwinPPM


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


def test_model():
    print("=" * 60)
    print("MultiTaskSwinPPM Model Test")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    in_channels = 1
    img_size = 256
    
    # Create model (pure PyTorch, no pretrained weights)
    print("\n[1] Creating model...")
    model = MultiTaskSwinPPM(in_channels=in_channels)
    
    # Input tensor
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"    Input shape: {x.shape}")
    
    # Forward pass
    print("\n[2] Running forward pass...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        lung_mask, infection_mask, class_logits = model(x)  # Uses default use_infection=True
        elapsed = time.time() - start_time
    
    print(f"    Forward time: {elapsed*1000:.2f} ms")
    
    # Output shapes
    print("\n[3] Output shapes:")
    print(f"    Lung mask:       {lung_mask.shape}")
    print(f"    Infection mask:  {infection_mask.shape}")
    print(f"    Class logits:    {class_logits.shape}")
    
    # Verify output shapes
    expected_mask_shape = (batch_size, 1, img_size, img_size)
    expected_cls_shape = (batch_size, 3)
    
    assert lung_mask.shape == expected_mask_shape, f"Lung mask shape mismatch: {lung_mask.shape}"
    assert infection_mask.shape == expected_mask_shape, f"Infection mask shape mismatch: {infection_mask.shape}"
    assert class_logits.shape == expected_cls_shape, f"Class logits shape mismatch: {class_logits.shape}"
    print("      All output shapes correct!")
    
    # Verify output ranges
    print("\n[4] Output value ranges:")
    print(f"    Lung mask:       min={lung_mask.min():.4f}, max={lung_mask.max():.4f}")
    print(f"    Infection mask:  min={infection_mask.min():.4f}, max={infection_mask.max():.4f}")
    print(f"    Class logits:    min={class_logits.min():.4f}, max={class_logits.max():.4f}")
    
    assert 0 <= lung_mask.min() and lung_mask.max() <= 1, "Lung mask should be in [0, 1]"
    assert 0 <= infection_mask.min() and infection_mask.max() <= 1, "Infection mask should be in [0, 1]"
    print("      Mask values in valid range [0, 1]!")
    
    # Parameter counts
    print("\n[5] Parameter counts:")
    print("-" * 40)
    
    encoder_params = count_parameters(model.encoder)
    ppm_params = count_parameters(model.ppm)
    decoder_params = count_parameters(model.decoder)
    lung_head_params = count_parameters(model.lung_head)
    inf_head_params = count_parameters(model.infection_head)
    cls_head_params = count_parameters(model.classification_head)
    total_params = count_parameters(model)
    
    print(f"    Swin Encoder:        {format_params(encoder_params):>10} ({encoder_params:>12,})")
    print(f"    PPM:                 {format_params(ppm_params):>10} ({ppm_params:>12,})")
    print(f"    Dual Decoder:        {format_params(decoder_params):>10} ({decoder_params:>12,})")
    print(f"    Lung Head:           {format_params(lung_head_params):>10} ({lung_head_params:>12,})")
    print(f"    Infection Head:      {format_params(inf_head_params):>10} ({inf_head_params:>12,})")
    print(f"    Classification Head: {format_params(cls_head_params):>10} ({cls_head_params:>12,})")
    print("-" * 40)
    print(f"    TOTAL:               {format_params(total_params):>10} ({total_params:>12,})")
    
    # Memory estimate
    print("\n[6] Memory estimate (FP32):")
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32, in MB
    print(f"    Parameters: {param_memory:.2f} MB")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n[7] CUDA test:")
        device = torch.device("cuda")
        model_cuda = model.to(device)
        x_cuda = x.to(device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model_cuda(x_cuda)
        torch.cuda.synchronize()
        elapsed_cuda = time.time() - start_time
        
        print(f"    CUDA forward time: {elapsed_cuda*1000:.2f} ms")
        print(f"    GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    else:
        print("\n[7] CUDA not available, skipping GPU test.")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_components():
    """Test individual components."""
    print("\n" + "=" * 60)
    print("Component Tests")
    print("=" * 60)
    
    from hybrid_model import SwinEncoder, PyramidPoolingModule, DualDecoder
    
    # Test encoder
    print("\n[A] SwinEncoder (Pure PyTorch):")
    encoder = SwinEncoder(in_channels=1)
    x = torch.randn(2, 1, 256, 256)
    features = encoder(x)
    for i, f in enumerate(features):
        print(f"    Stage {i+1}: {f.shape}")
    
    # Test PPM
    print("\n[B] PyramidPoolingModule:")
    ppm = PyramidPoolingModule(in_channels=768)
    x = torch.randn(2, 768, 8, 8)
    out = ppm(x)
    print(f"    Input:  {x.shape}")
    print(f"    Output: {out.shape}")
    
    # Test decoder
    print("\n[C] DualDecoder:")
    decoder = DualDecoder()
    bottleneck = torch.randn(2, 768, 8, 8)
    encoder_features = [
        torch.randn(2, 96, 64, 64),
        torch.randn(2, 192, 32, 32),
        torch.randn(2, 384, 16, 16),
        torch.randn(2, 768, 8, 8),
    ]
    lung, inf = decoder(bottleneck, encoder_features)
    print(f"    Bottleneck: {bottleneck.shape}")
    print(f"    Lung output: {lung.shape}")
    print(f"    Infection output: {inf.shape}")
    
    print("\n   All component tests passed!")


if __name__ == "__main__":
    test_model()
    test_components()