# Phase 2: Test on Infection Segmentation (Test Set)
# Evaluate the trained model on the test set

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for hybrid_model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import MultiTaskSwinPPM, dice_score, iou_score


class InfectionSegmentationDataset(Dataset):
    """
    Dataset for Infection Segmentation Data (Test mode with filenames).
    """
    
    CLASS_MAP = {'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}
    
    def __init__(self, data_root, split='Test', img_size=(256, 256)):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        self.samples = []
        
        split_dir = os.path.join(data_root, 'Infection Segmentation Data', 
                                  'Infection Segmentation Data', split)
        
        for class_name in ['COVID-19', 'Non-COVID', 'Normal']:
            class_dir = os.path.join(split_dir, class_name)
            images_dir = os.path.join(class_dir, 'images')
            lung_masks_dir = os.path.join(class_dir, 'lung masks')
            infection_masks_dir = os.path.join(class_dir, 'infection masks')
            has_infection_masks = os.path.exists(infection_masks_dir) and class_name == 'COVID-19'
            
            if not os.path.exists(images_dir):
                continue
            
            for img_name in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_name)
                lung_mask_path = os.path.join(lung_masks_dir, img_name)
                
                if has_infection_masks:
                    infection_mask_path = os.path.join(infection_masks_dir, img_name)
                else:
                    infection_mask_path = None
                
                if os.path.exists(img_path) and os.path.exists(lung_mask_path):
                    self.samples.append({
                        'img_path': img_path,
                        'lung_mask_path': lung_mask_path,
                        'infection_mask_path': infection_mask_path,
                        'class_name': class_name,
                        'label': self.CLASS_MAP[class_name],
                        'filename': img_name
                    })
        
        print(f"Loaded {len(self.samples)} samples from {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(sample['img_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        
        # Load lung mask
        lung_mask = cv2.imread(sample['lung_mask_path'], cv2.IMREAD_GRAYSCALE)
        lung_mask = cv2.resize(lung_mask, self.img_size)
        lung_mask = (lung_mask > 127).astype(np.float32)
        lung_mask = torch.from_numpy(lung_mask).unsqueeze(0)
        
        # Load infection mask
        if sample['infection_mask_path'] and os.path.exists(sample['infection_mask_path']):
            inf_mask = cv2.imread(sample['infection_mask_path'], cv2.IMREAD_GRAYSCALE)
            inf_mask = cv2.resize(inf_mask, self.img_size)
            inf_mask = (inf_mask > 127).astype(np.float32)
        else:
            inf_mask = np.zeros(self.img_size, dtype=np.float32)
        inf_mask = torch.from_numpy(inf_mask).unsqueeze(0)
        
        return img, lung_mask, inf_mask, sample['label'], sample['filename'], sample['class_name']


def visualize_predictions(model, dataset, device, save_dir, num_samples=8):
    """Visualize predictions on sample images."""
    model.eval()
    
    # Get COVID-19 samples (they have infection masks)
    covid_indices = [i for i, s in enumerate(dataset.samples) if s['class_name'] == 'COVID-19']
    indices = covid_indices[:num_samples] if len(covid_indices) >= num_samples else covid_indices
    
    fig, axes = plt.subplots(len(indices), 5, figsize=(20, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, lung_mask, inf_mask, label, filename, class_name = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            
            lung_pred, inf_pred, class_pred = model(img_tensor, use_infection=True)
            lung_pred = lung_pred.squeeze().cpu().numpy()
            inf_pred = inf_pred.squeeze().cpu().numpy()
            
            img_np = img.squeeze().numpy()
            lung_gt = lung_mask.squeeze().numpy()
            inf_gt = inf_mask.squeeze().numpy()
            
            # Image
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f'Image\n{filename[:20]}...')
            axes[i, 0].axis('off')
            
            # Lung GT vs Pred
            axes[i, 1].imshow(lung_gt, cmap='gray')
            axes[i, 1].set_title(f'Lung GT')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow((lung_pred > 0.5).astype(float), cmap='gray')
            lung_dice = dice_score(torch.tensor(lung_pred).unsqueeze(0).unsqueeze(0), 
                                   lung_mask.unsqueeze(0))
            axes[i, 2].set_title(f'Lung Pred\nDice: {lung_dice:.4f}')
            axes[i, 2].axis('off')
            
            # Infection GT vs Pred
            axes[i, 3].imshow(inf_gt, cmap='Reds')
            axes[i, 3].set_title(f'Infection GT')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow((inf_pred > 0.5).astype(float), cmap='Reds')
            inf_dice = dice_score(torch.tensor(inf_pred).unsqueeze(0).unsqueeze(0), 
                                  inf_mask.unsqueeze(0))
            axes[i, 4].set_title(f'Infection Pred\nDice: {inf_dice:.4f}')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def evaluate_model(model, loader, device):
    """Evaluate model and return per-sample metrics."""
    model.eval()
    
    results = []
    total_lung_dice = 0
    total_lung_iou = 0
    total_inf_dice = 0
    total_inf_iou = 0
    total_correct = 0
    total_samples = 0
    
    class_lung_dice = {'Normal': [], 'Non-COVID': [], 'COVID-19': []}
    class_inf_dice = {'Normal': [], 'Non-COVID': [], 'COVID-19': []}
    class_correct = {'Normal': 0, 'Non-COVID': 0, 'COVID-19': 0}
    class_total = {'Normal': 0, 'Non-COVID': 0, 'COVID-19': 0}
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, lung_masks, inf_masks, labels, filenames, class_names in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            lung_masks = lung_masks.to(device)
            inf_masks = inf_masks.to(device)
            labels_tensor = torch.tensor([{'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}[cn] for cn in class_names]).to(device)
            
            lung_pred, inf_pred, class_pred = model(images, use_infection=True)
            
            pred_labels = class_pred.argmax(dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())
            
            for j in range(images.size(0)):
                lung_d = dice_score(lung_pred[j:j+1], lung_masks[j:j+1])
                lung_iou = iou_score(lung_pred[j:j+1], lung_masks[j:j+1])
                inf_d = dice_score(inf_pred[j:j+1], inf_masks[j:j+1])
                inf_iou = iou_score(inf_pred[j:j+1], inf_masks[j:j+1])
                
                total_lung_dice += lung_d
                total_lung_iou += lung_iou
                total_inf_dice += inf_d
                total_inf_iou += inf_iou
                
                cn = class_names[j]
                class_lung_dice[cn].append(lung_d)
                class_inf_dice[cn].append(inf_d)
                
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
                    'lung_dice': lung_d,
                    'lung_iou': lung_iou,
                    'inf_dice': inf_d,
                    'inf_iou': inf_iou
                })
    
    n = len(loader.dataset)
    covid_count = sum(1 for s in loader.dataset.samples if s['class_name'] == 'COVID-19')
    avg_lung_dice = total_lung_dice / n
    avg_lung_iou = total_lung_iou / n
    # Infection metrics only for COVID-19 cases
    avg_inf_dice = total_inf_dice / covid_count if covid_count > 0 else 0
    avg_inf_iou = total_inf_iou / covid_count if covid_count > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    # Per-class metrics
    class_metrics = {}
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_names_list = ['Normal', 'Non-COVID', 'COVID-19']
    for i, cn in enumerate(class_names_list):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cn] = {
            'lung_dice': np.mean(class_lung_dice[cn]) if class_lung_dice[cn] else 0,
            'inf_dice': np.mean(class_inf_dice[cn]) if class_inf_dice[cn] else 0,
            'acc': class_correct[cn] / class_total[cn] if class_total[cn] > 0 else 0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(class_lung_dice[cn])
        }
    
    # Macro metrics
    macro_precision = np.mean([class_metrics[cn]['precision'] for cn in class_names_list])
    macro_recall = np.mean([class_metrics[cn]['recall'] for cn in class_names_list])
    macro_f1 = np.mean([class_metrics[cn]['f1'] for cn in class_names_list])
    
    # Confusion matrix
    conf_matrix = np.zeros((3, 3), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        conf_matrix[label, pred] += 1
    
    return (avg_lung_dice, avg_lung_iou, avg_inf_dice, avg_inf_iou, avg_acc, 
            macro_precision, macro_recall, macro_f1, results, class_metrics, conf_matrix)


def main():
    # ============== CONFIG ==============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'archive')
    
    model_path = os.path.join(script_dir, 'phase2_best_model.pth')
    results_dir = os.path.join(script_dir, 'phase2_results')
    os.makedirs(results_dir, exist_ok=True)
    
    img_size = (256, 256)
    batch_size = 16
    
    print("=" * 80)
    print("Phase 2: Infection Segmentation Testing")
    print("=" * 80)
    
    # ============== DATASET ==============
    test_dataset = InfectionSegmentationDataset(data_root, split='Test', img_size=img_size)
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
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train2.py first to train the model.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  val_lung_dice: {checkpoint.get('val_lung_dice', 'N/A')}")
    print(f"  val_inf_dice: {checkpoint.get('val_inf_dice', 'N/A')}")
    print(f"  val_acc: {checkpoint.get('val_acc', 'N/A')}")
    
    # ============== EVALUATION ==============
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)
    (avg_lung_dice, avg_lung_iou, avg_inf_dice, avg_inf_iou, avg_acc,
     macro_prec, macro_rec, macro_f1, results, class_metrics, conf_matrix) = metrics
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    print(f"\n[Segmentation Metrics]")
    print(f"  Lung Dice:      {avg_lung_dice:.4f}")
    print(f"  Lung IoU:       {avg_lung_iou:.4f}")
    print(f"  Infection Dice: {avg_inf_dice:.4f}")
    print(f"  Infection IoU:  {avg_inf_iou:.4f}")
    
    print(f"\n[Classification Metrics]")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {macro_prec:.4f} (macro)")
    print(f"  Recall:    {macro_rec:.4f} (macro)")
    print(f"  F1 Score:  {macro_f1:.4f} (macro)")
    
    print(f"\n[Per-Class Metrics]")
    print("-" * 100)
    print(f"{'Class':<12} {'Count':<8} {'L.Dice':<10} {'I.Dice':<10} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 100)
    for cn in ['Normal', 'Non-COVID', 'COVID-19']:
        m = class_metrics[cn]
        print(f"{cn:<12} {m['count']:<8} {m['lung_dice']:.4f}     {m['inf_dice']:.4f}     "
              f"{m['acc']:.4f}   {m['precision']:.4f}   {m['recall']:.4f}   {m['f1']:.4f}")
    print("-" * 100)
    
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
        f.write("Phase 2: Full Multi-Task Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n\n")
        
        f.write("Segmentation Metrics:\n")
        f.write(f"  Lung Dice:      {avg_lung_dice:.4f}\n")
        f.write(f"  Lung IoU:       {avg_lung_iou:.4f}\n")
        f.write(f"  Infection Dice: {avg_inf_dice:.4f}\n")
        f.write(f"  Infection IoU:  {avg_inf_iou:.4f}\n\n")
        
        f.write("Classification Metrics:\n")
        f.write(f"  Accuracy:  {avg_acc:.4f}\n")
        f.write(f"  Precision: {macro_prec:.4f} (macro)\n")
        f.write(f"  Recall:    {macro_rec:.4f} (macro)\n")
        f.write(f"  F1 Score:  {macro_f1:.4f} (macro)\n\n")
        
        f.write("Per-Class Metrics:\n")
        for cn in ['Normal', 'Non-COVID', 'COVID-19']:
            m = class_metrics[cn]
            f.write(f"  {cn}: L.Dice={m['lung_dice']:.4f}, I.Dice={m['inf_dice']:.4f}, "
                    f"P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, n={m['count']}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"              Normal    Non-COVID  COVID-19\n")
        for i, cn in enumerate(['Normal', 'Non-COVID', 'COVID-19']):
            f.write(f"  {cn:<10} {conf_matrix[i, 0]:<10} {conf_matrix[i, 1]:<10} {conf_matrix[i, 2]:<10}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Normal', 'Non-COVID', 'COVID-19'])
    ax.set_yticklabels(['Normal', 'Non-COVID', 'COVID-19'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    for i in range(3):
        for j in range(3):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', 
                   color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black', fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    conf_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_path}")
    
    # Visualization
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_dataset, device, results_dir, num_samples=8)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
