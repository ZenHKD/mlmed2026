import os
import torch
import cv2
import pandas as pd
import numpy as np
from model import CU_Net
import matplotlib.pyplot as plt
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
        raise FileNotFoundError(f"Could not open {img_path}")
        
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
    data_dir = os.path.join(script_dir, 'data/training_set')
    csv_path = os.path.join(script_dir, 'data/training_set_pixel_size_and_HC.csv')
    model_path = os.path.join(script_dir, 'best_model.pth')
    split_path = os.path.join(script_dir, 'data_split.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not os.path.exists(split_path):
        print(f"Error: Data split file not found at {split_path}")
        print("Please run train.py first to generate the split.")
        return

    # Load Full Data
    df = pd.read_csv(csv_path)
    
    # Load Split
    df_split = pd.read_csv(split_path)
    val_filenames = df_split['val'].dropna().tolist()
    
    print(f"Loaded split. Found {len(val_filenames)} validation images.")
    
    # Filter DataFrame for Validation Set
    val_df = df[df['filename'].isin(val_filenames)]
    
    if len(val_df) == 0:
        print("Warning: No validation images found in the main CSV. Check filenames.")
        return
        
    print(f"Processing {len(val_df)} validation images...")
    
    # Load model
    model = CU_Net(in_channels=1, num_classes=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
        
    predictions = []
    ground_truths = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        filename = row['filename']
        pixel_size = row['pixel size(mm)']
        gt_hc = row['head circumference (mm)']
        
        img_path = os.path.join(data_dir, filename)
        
        # Predict
        try:
            mask = predict_mask(model, img_path, device)
            pred_hc = get_hc_from_mask(mask, pixel_size)
            
            predictions.append(pred_hc)
            ground_truths.append(gt_hc)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            predictions.append(0)
            ground_truths.append(gt_hc)

    # Analyze results
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Filter out failed predictions (0.0)
    valid_mask = predictions > 0
    valid_preds = predictions[valid_mask]
    valid_gts = ground_truths[valid_mask]
    
    if len(valid_preds) == 0:
        print("No valid predictions made.")
        return

    mae = np.mean(np.abs(valid_preds - valid_gts))
    mse = np.mean((valid_preds - valid_gts) ** 2)
    rmse = np.sqrt(mse)
    
    print("\n--- Validation Results ---")
    print(f"Total processed: {len(predictions)}")
    print(f"Valid predictions: {len(valid_preds)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
    
    # R-squared
    ss_res = np.sum((valid_gts - valid_preds) ** 2)
    ss_tot = np.sum((valid_gts - np.mean(valid_gts)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r2:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(valid_gts, valid_preds, alpha=0.5)
    plt.plot([min(valid_gts), max(valid_gts)], [min(valid_gts), max(valid_gts)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth HC (mm)')
    plt.ylabel('Predicted HC (mm)')
    plt.title(f'Validation Set Regression Analysis\nMAE: {mae:.2f}mm, R2: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
