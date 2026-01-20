import os
import torch
import cv2
import numpy as np
import pandas as pd
from model import CU_Net


def predict(model, img_path, device, img_size=(512, 512)):
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


def main():
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
    
    for idx, filename in enumerate(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, filename)
        
        # Predict mask
        mask = predict(model, img_path, device)
        
        # Save predicted mask
        mask_filename = filename.replace('.png', '_pred.png')
        cv2.imwrite(os.path.join(output_dir, mask_filename), (mask * 255).astype(np.uint8))
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(os.listdir(test_dir))} images")
    
    # Save results
    print(f"Predicted masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
