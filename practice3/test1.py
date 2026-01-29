# Phase 1: Test on Lung Segmentation (Test Set)
# Evaluate the trained model on the test set

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for hybrid_model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import MultiTaskSwinPPM, dice_score, iou_score


class LungSegmentationDataset(Dataset):
    """
    Dataset for Lung Segmentation Data (Test split).
    """
    
    CLASS_MAP = {'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}
    
    def __init__(self, data_root, split='Test', img_size=(256, 256)):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        # Collect all image paths
        self.samples = []
        
        split_dir = os.path.join(data_root, 'Lung Segmentation Data', 'Lung Segmentation Data', split)
        
        for class_name in ['COVID-19', 'Non-COVID', 'Normal']:
            class_dir = os.path.join(split_dir, class_name)
            images_dir = os.path.join(class_dir, 'images')
            masks_dir = os.path.join(class_dir, 'lung masks')
            
            if not os.path.exists(images_dir):
                continue
                
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(images_dir, img_file)
                    mask_path = os.path.join(masks_dir, img_file)
                    
                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image': img_path,
                            'mask': mask_path,
                            'class': self.CLASS_MAP[class_name],
                            'class_name': class_name,
                            'filename': img_file
                        })
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (grayscale)
        img = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        
        # Load lung mask
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)  # (1, H, W)
        
        # Class label
        label = sample['class']
        
        return img, mask, label, sample['filename'], sample['class_name']


def visualize_predictions(model, dataset, device, save_dir, num_samples=12):
    """Visualize predictions on random samples."""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Random sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, label, filename, class_name = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            lung_pred, inf_pred, class_pred = model(img, use_infection=False)
            lung_pred = lung_pred.squeeze().cpu().numpy()
            
            img_np = img.squeeze().cpu().numpy()
            mask_np = mask.squeeze().numpy()
            
            # Plot
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f'{class_name}\n{filename}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(lung_pred, cmap='gray')
            axes[i, 2].set_title('Prediction (Raw)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow((lung_pred > 0.5).astype(float), cmap='gray')
            dice = dice_score(torch.tensor(lung_pred).unsqueeze(0).unsqueeze(0), 
                             mask.unsqueeze(0))
            axes[i, 3].set_title(f'Prediction (Binary)\nDice: {dice:.4f}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def evaluate_model(model, loader, device):
    """Evaluate model and return per-sample metrics including classification."""
    model.eval()
    
    results = []
    total_dice = 0
    total_iou = 0
    total_correct = 0
    total_samples = 0
    
    class_dice = {'Normal': [], 'Non-COVID': [], 'COVID-19': []}
    class_iou = {'Normal': [], 'Non-COVID': [], 'COVID-19': []}
    class_correct = {'Normal': 0, 'Non-COVID': 0, 'COVID-19': 0}
    class_total = {'Normal': 0, 'Non-COVID': 0, 'COVID-19': 0}
    
    # Confusion matrix data
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, labels, filenames, class_names in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            labels_tensor = torch.tensor([{'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}[cn] for cn in class_names]).to(device)
            
            lung_pred, inf_pred, class_pred = model(images, use_infection=False)
            
            # Classification predictions
            pred_labels = class_pred.argmax(dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())
            
            # Per-sample metrics
            for j in range(images.size(0)):
                d = dice_score(lung_pred[j:j+1], masks[j:j+1])
                iou = iou_score(lung_pred[j:j+1], masks[j:j+1])
                
                total_dice += d
                total_iou += iou
                
                cn = class_names[j]
                class_dice[cn].append(d)
                class_iou[cn].append(iou)
                
                # Classification accuracy
                is_correct = pred_labels[j].item() == labels_tensor[j].item()
                if is_correct:
                    class_correct[cn] += 1
                    total_correct += 1
                class_total[cn] += 1
                total_samples += 1
                
                results.append({
                    'filename': filenames[j],
                    'class': cn,
                    'pred_class': ['Normal', 'Non-COVID', 'COVID-19'][pred_labels[j].item()],
                    'correct': is_correct,
                    'dice': d,
                    'iou': iou
                })
    
    n = len(loader.dataset)
    avg_dice = total_dice / n
    avg_iou = total_iou / n
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    # Per-class averages
    class_metrics = {}
    for cn in ['Normal', 'Non-COVID', 'COVID-19']:
        class_metrics[cn] = {
            'dice': np.mean(class_dice[cn]) if class_dice[cn] else 0,
            'iou': np.mean(class_iou[cn]) if class_iou[cn] else 0,
            'acc': class_correct[cn] / class_total[cn] if class_total[cn] > 0 else 0,
            'count': len(class_dice[cn])
        }
    
    # Convert to numpy arrays for sklearn
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute precision, recall, F1 per class
    class_names_list = ['Normal', 'Non-COVID', 'COVID-19']
    for i, cn in enumerate(class_names_list):
        # True positives, false positives, false negatives
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cn]['precision'] = precision
        class_metrics[cn]['recall'] = recall
        class_metrics[cn]['f1'] = f1
    
    # Macro-averaged metrics
    macro_precision = np.mean([class_metrics[cn]['precision'] for cn in class_names_list])
    macro_recall = np.mean([class_metrics[cn]['recall'] for cn in class_names_list])
    macro_f1 = np.mean([class_metrics[cn]['f1'] for cn in class_names_list])
    
    # Confusion matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label, pred] += 1
    
    return avg_dice, avg_iou, avg_acc, macro_precision, macro_recall, macro_f1, results, class_metrics, confusion_matrix


def main():
    # ============== CONFIG ==============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'archive')
    
    # Paths
    model_path = os.path.join(script_dir, 'phase1_best_model.pth')
    results_dir = os.path.join(script_dir, 'phase1_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Hyperparameters
    img_size = (256, 256)
    batch_size = 16
    
    print("=" * 80)
    print("Phase 1: Lung Segmentation Testing")
    print("=" * 80)
    
    # ============== DATASET ==============
    test_dataset = LungSegmentationDataset(data_root, split='Test', img_size=img_size)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    print("-" * 80)
    
    # ============== MODEL ==============
    model = MultiTaskSwinPPM(in_channels=1, num_classes=3).to(device)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train1.py first to train the model.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_dice: {checkpoint['val_dice']:.4f})")
    
    # ============== EVALUATION ==============
    print("\nEvaluating on test set...")
    avg_dice, avg_iou, avg_acc, macro_prec, macro_rec, macro_f1, results, class_metrics, conf_matrix = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    # Segmentation Metrics
    print(f"\n[Segmentation Metrics]")
    print(f"  Dice Score: {avg_dice:.4f}")
    print(f"  IoU Score:  {avg_iou:.4f}")
    
    # Classification Metrics
    print(f"\n[Classification Metrics]")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {macro_prec:.4f} (macro)")
    print(f"  Recall:    {macro_rec:.4f} (macro)")
    print(f"  F1 Score:  {macro_f1:.4f} (macro)")
    
    # Per-Class Metrics Table
    print(f"\n[Per-Class Metrics]")
    print("-" * 90)
    print(f"{'Class':<12} {'Count':<8} {'Dice':<8} {'IoU':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 90)
    for cn in ['Normal', 'Non-COVID', 'COVID-19']:
        m = class_metrics[cn]
        print(f"{cn:<12} {m['count']:<8} {m['dice']:.4f}   {m['iou']:.4f}   {m['acc']:.4f}   "
              f"{m['precision']:.4f}   {m['recall']:.4f}   {m['f1']:.4f}")
    print("-" * 90)
    
    # Confusion Matrix
    print(f"\n[Confusion Matrix]")
    print("                  Predicted")
    print(f"              {'Normal':<12} {'Non-COVID':<12} {'COVID-19':<12}")
    for i, cn in enumerate(['Normal', 'Non-COVID', 'COVID-19']):
        print(f"  {cn:<10} {conf_matrix[i, 0]:<12} {conf_matrix[i, 1]:<12} {conf_matrix[i, 2]:<12}")
    
    # Save results
    results_path = os.path.join(results_dir, 'test_results.csv')
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nPer-sample results saved to: {results_path}")
    
    # Summary
    summary_path = os.path.join(results_dir, 'test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Phase 1: Lung Segmentation + Classification Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Val Dice (at save): {checkpoint['val_dice']:.4f}\n")
        f.write(f"Val Acc (at save): {checkpoint.get('val_acc', 'N/A')}\n\n")
        
        f.write("Segmentation Metrics:\n")
        f.write(f"  Dice: {avg_dice:.4f}\n")
        f.write(f"  IoU:  {avg_iou:.4f}\n\n")
        
        f.write("Classification Metrics:\n")
        f.write(f"  Accuracy:  {avg_acc:.4f}\n")
        f.write(f"  Precision: {macro_prec:.4f} (macro)\n")
        f.write(f"  Recall:    {macro_rec:.4f} (macro)\n")
        f.write(f"  F1 Score:  {macro_f1:.4f} (macro)\n\n")
        
        f.write("Per-Class Metrics:\n")
        for cn in ['Normal', 'Non-COVID', 'COVID-19']:
            m = class_metrics[cn]
            f.write(f"  {cn}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, "
                    f"P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, n={m['count']}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"              Normal    Non-COVID  COVID-19\n")
        for i, cn in enumerate(['Normal', 'Non-COVID', 'COVID-19']):
            f.write(f"  {cn:<10} {conf_matrix[i, 0]:<10} {conf_matrix[i, 1]:<10} {conf_matrix[i, 2]:<10}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Save confusion matrix as image
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Normal', 'Non-COVID', 'COVID-19'])
    ax.set_yticklabels(['Normal', 'Non-COVID', 'COVID-19'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, conf_matrix[i, j], ha='center', va='center', 
                          color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black', fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    conf_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_path}")
    
    # ============== VISUALIZATION ==============
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_dataset, device, results_dir, num_samples=8)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
