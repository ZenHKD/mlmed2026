import os
import torch
import pandas as pd
from model import CU_Net
from tqdm import tqdm
from val_reg import predict_mask, get_hc_from_mask


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
