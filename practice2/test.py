import os
import torch
import cv2
import numpy as np
from model import CU_Net


def predict(model, img_path, device, img_size=(256, 256)):
    """Predict segmentation mask for a single image"""
    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, img_size)
    img_tensor = torch.tensor(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Predict
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Resize back to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]))
    return mask


def create_overlay(img_path, mask):
    """Create overlay image with prediction mask on original image (green overlay)"""
    # Load original image in color
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Binarize mask and create alpha channel
    mask_binary = (mask > 0.5).astype(np.float32)
    alpha = mask_binary[..., np.newaxis]  # expand to (H, W, 1) for broadcasting
    
    # Green overlay color
    green = np.array([0, 255, 0], dtype=np.float32)  # RGB green
    
    # Blend: where mask is 1, mix 60% image + 40% green
    overlay = (1 - alpha) * img_rgb + alpha * (0.6 * img_rgb + 0.4 * green)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for cv2.imwrite
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return overlay_bgr


def main():
    """Run inference on test set and save predictions with overlays."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, 'data/test_set')
    model_path = os.path.join(script_dir, 'best_model.pth')
    output_dir = os.path.join(script_dir, 'predictions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = CU_Net(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    filenames = sorted(os.listdir(test_dir))
    for idx, filename in enumerate(filenames):
        img_path = os.path.join(test_dir, filename)
        
        # Predict mask
        mask = predict(model, img_path, device)
        
        # Save predicted mask
        mask_filename = filename.replace('.png', '_pred.png')
        cv2.imwrite(os.path.join(output_dir, mask_filename), (mask * 255).astype(np.uint8))
        
        # Save overlay image
        overlay = create_overlay(img_path, mask)
        overlay_filename = filename.replace('.png', '_overlay.png')
        cv2.imwrite(os.path.join(output_dir, overlay_filename), overlay)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(filenames)} images")
    
    print(f"\nPredicted masks and overlays saved to: {output_dir}")


if __name__ == "__main__":
    main()
