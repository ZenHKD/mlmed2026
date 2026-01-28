import os
import torch
import cv2
import pandas as pd
import numpy as np
from model import CU_Net
from tqdm import tqdm

def calculate_ellipse_perimeter(a, b):
    """
    Calculate the perimeter of an ellipse using Ramanujan's approximation.
    a: semi-major axis (width/2)
    b: semi-minor axis (height/2)
    """
    # Ramanujan's second approximation
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    return perimeter

def predict_mask(model, img_path, device, img_size=(256, 256)):
    """Predict segmentation mask for a single image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Colud not open {img_path}")
        
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, img_size)
    img_tensor = torch.tensor(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        mask_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Resize back to original size
    mask_prob = cv2.resize(mask_prob, (original_size[1], original_size[0]))
    return mask_prob

def get_hc_from_mask(mask, pixel_size):
    """
    Calculate Head Circumference (HC) from segmentation mask.
    """
    # Threshold mask
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Get largest contour (assuming it's the head)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Need at least 5 points to fit an ellipse
    if len(largest_contour) < 5:
        return 0.0
        
    try:
        # Fit ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
        
        # MA/ma are Major/minor Axis Diameters -> convert to semi-axes (radius)
        a = MA / 2
        b = ma / 2
        
        # Calculate perimeter in pixels
        perimeter_pixels = calculate_ellipse_perimeter(a, b)
        
        # Convert to mm
        hc_mm = perimeter_pixels * pixel_size
        return hc_mm
        
    except Exception as e:
        print(f"Error fitting ellipse: {e}")
        return 0.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data/test_set')
    csv_path = os.path.join(script_dir, 'data/test_set_pixel_size.csv')
    model_path = os.path.join(script_dir, 'best_model.pth')
    output_csv_path = os.path.join(script_dir, 'test_set_HC.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded test CSV with {len(df)} entries")
    
    # Load model
    model = CU_Net(in_channels=1, num_classes=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
        
    results = []
    
    print("Starting prediction on test set...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        pixel_size = row['pixel size(mm)']
        
        img_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {filename} not found.")
            results.append({'filename': filename, 'head circumference (mm)': 0.0})
            continue

        try:
            mask = predict_mask(model, img_path, device)
            pred_hc = get_hc_from_mask(mask, pixel_size)
            
            results.append({
                'filename': filename,
                'head circumference (mm)': round(pred_hc, 2)
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({'filename': filename, 'head circumference (mm)': 0.0})

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")

if __name__ == "__main__":
    main()
