"""
LUNA16 Validation:
  Stage 1: Lung segmentation metrics (Dice, IoU)
  Stage 2: Classification metrics (Accuracy, Precision, Recall, F1, AUC)
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score


def validate_seg(model, dataloader, criterion, device):
    """Validate lung segmentation model (Stage 1).
    
    Returns:
        avg_loss, metrics_dict
    """
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Val-Seg', leave=False)
        for batch in pbar:
            image = batch['image'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(image)
                loss_dict = criterion(pred, mask)
            
            total_loss += loss_dict['total_loss'].item()
            
            # Compute Dice and IoU
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            intersection = (pred_bin * mask).sum()
            union = pred_bin.sum() + mask.sum()
            
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            
            total_dice += dice.item()
            total_iou += iou.item()
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    
    metrics = {
        "lung_Dice": avg_dice,
        "lung_IoU": avg_iou,
    }
    
    return avg_loss, metrics


def validate_cls(model, dataloader, criterion, device):
    """Validate candidate classification model (Stage 2).
    
    Returns:
        avg_loss, metrics_dict
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Val-Cls', leave=False)
        for batch in pbar:
            patch = batch['patch'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(patch)
                loss_dict = criterion(pred, label)
            
            total_loss += loss_dict['total_loss'].item()
            
            probs = torch.sigmoid(pred).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = label.cpu().numpy().astype(int)
            
            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.flatten().tolist())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    # Sensitivity at various FP rates (useful for FROC)
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }
    
    return avg_loss, metrics
