# Phase 1: Train on Lung Segmentation (Full Dataset)
# Train only the lung segmentation branch

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for hybrid_model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import MultiTaskSwinPPM, BCEDiceLoss, dice_score, iou_score


class LungSegmentationDataset(Dataset):
    """
    Dataset for Lung Segmentation Data.
    
    Structure:
        archive/Lung Segmentation Data/Lung Segmentation Data/
            Train/
                COVID-19/images/, COVID-19/lung masks/
                Non-COVID/images/, Non-COVID/lung masks/
                Normal/images/, Normal/lung masks/
            Val/
                ...
            Test/
                ...
    """
    
    CLASS_MAP = {'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}
    
    def __init__(self, data_root, split='Train', img_size=(256, 256)):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        # Collect all image paths and corresponding mask paths
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
                            'class_name': class_name
                        })
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Class distribution
        class_counts = {}
        for s in self.samples:
            cn = s['class_name']
            class_counts[cn] = class_counts.get(cn, 0) + 1
        for cn, count in class_counts.items():
            print(f"  {cn}: {count}")
    
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
        mask = (mask > 127).astype(np.float32)  # Binarize
        mask = torch.tensor(mask).unsqueeze(0)  # (1, H, W)
        
        # Class label
        label = sample['class']
        
        return img, mask, label


def train_epoch(model, loader, seg_criterion, cls_criterion, optimizer, device, 
                 seg_weight=1.0, cls_weight=0.5):
    """Train for one epoch with lung segmentation + classification."""
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_cls_loss = 0
    total_dice = 0
    total_iou = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, masks, labels in pbar:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        lung_pred, inf_pred, class_pred = model(images, use_infection=False)
        
        # Segmentation loss (lung only)
        seg_loss = seg_criterion(lung_pred, masks)
        
        # Classification loss
        cls_loss = cls_criterion(class_pred, labels)
        
        # Combined loss
        loss = seg_weight * seg_loss + cls_weight * cls_loss
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_dice += dice_score(lung_pred, masks)
        total_iou += iou_score(lung_pred, masks)
        
        # Classification accuracy
        pred_labels = class_pred.argmax(dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0
    return total_loss / n, total_seg_loss / n, total_cls_loss / n, total_dice / n, total_iou / n, acc


def evaluate(model, loader, seg_criterion, cls_criterion, device, 
             seg_weight=1.0, cls_weight=0.5):
    """Evaluate model on validation set with segmentation + classification."""
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_cls_loss = 0
    total_dice = 0
    total_iou = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            lung_pred, inf_pred, class_pred = model(images, use_infection=False)
            
            # Segmentation loss
            seg_loss = seg_criterion(lung_pred, masks)
            
            # Classification loss
            cls_loss = cls_criterion(class_pred, labels)
            
            # Combined loss
            loss = seg_weight * seg_loss + cls_weight * cls_loss
            
            # Metrics
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_dice += dice_score(lung_pred, masks)
            total_iou += iou_score(lung_pred, masks)
            
            # Classification accuracy
            pred_labels = class_pred.argmax(dim=1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += labels.size(0)
    
    n = len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0
    return total_loss / n, total_seg_loss / n, total_cls_loss / n, total_dice / n, total_iou / n, acc


def main():
    # ============== CONFIG ==============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'archive')
    
    # Paths
    best_model_path = os.path.join(script_dir, 'phase1_best_model.pth')
    last_model_path = os.path.join(script_dir, 'phase1_last_model.pth')
    log_path = os.path.join(script_dir, 'phase1_training_log.csv')
    
    # Hyperparameters
    img_size = (256, 256)
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    
    print("=" * 80)
    print("Phase 1: Lung Segmentation Training")
    print("=" * 80)
    
    # ============== DATASET ==============
    train_dataset = LungSegmentationDataset(data_root, split='Train', img_size=img_size)
    val_dataset = LungSegmentationDataset(data_root, split='Val', img_size=img_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {img_size}, Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}, Learning rate: {learning_rate}")
    print("-" * 80)
    
    # ============== MODEL ==============
    model = MultiTaskSwinPPM(in_channels=1, num_classes=3).to(device)
    
    # ============== FREEZE LAYERS (Phase 1) ==============
    # Freeze infection decoder and infection head only (not trained in Phase 1)
    for param in model.decoder.inf_blocks.parameters():
        param.requires_grad = False
    for param in model.decoder.inf_cross_attn.parameters():
        param.requires_grad = False
    for param in model.infection_head.parameters():
        param.requires_grad = False
    
    print("\nPhase 1 Layer Freezing:")
    print("  [FROZEN] Infection decoder blocks")
    print("  [FROZEN] Infection cross-attention")
    print("  [FROZEN] Infection head")
    print("  [TRAINABLE] Encoder, PPM, Lung decoder, Lung head, Classification head")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    
    # ============== LOSS & OPTIMIZER ==============
    seg_criterion = BCEDiceLoss(bce_weight=0.5)
    cls_criterion = torch.nn.CrossEntropyLoss()
    seg_weight = 1.0
    cls_weight = 0.5
    
    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # ============== TRAINING LOOP ==============
    best_val_dice = 0.0
    training_history = []
    
    print("\nStarting training...")
    print("-" * 100)
    
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_seg, train_cls, train_dice, train_iou, train_acc = train_epoch(
            model, train_loader, seg_criterion, cls_criterion, 
            optimizer, device, seg_weight, cls_weight
        )
        
        # Validate
        val_loss, val_seg, val_cls, val_dice, val_iou, val_acc = evaluate(
            model, val_loader, seg_criterion, cls_criterion, device, seg_weight, cls_weight
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_loss,
            'train_seg_loss': train_seg,
            'train_cls_loss': train_cls,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_seg_loss': val_seg,
            'val_cls_loss': val_cls,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_acc': val_acc
        }
        training_history.append(epoch_log)
        
        # Save training log
        pd.DataFrame(training_history).to_csv(log_path, index=False)
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_dice': val_dice,
            'val_acc': val_acc,
        }, last_model_path)
        
        # Save best model (based on val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_acc': val_acc,
            }, best_model_path)
            print(f'Epoch {epoch:03d}  |  lr {current_lr:.2e}  |  '
                  f'train: loss {train_loss:.4f} dice {train_dice:.4f} acc {train_acc:.4f}  |  '
                  f'val: loss {val_loss:.4f} dice {val_dice:.4f} acc {val_acc:.4f}  |  *** Best ***')
        else:
            print(f'Epoch {epoch:03d}  |  lr {current_lr:.2e}  |  '
                  f'train: loss {train_loss:.4f} dice {train_dice:.4f} acc {train_acc:.4f}  |  '
                  f'val: loss {val_loss:.4f} dice {val_dice:.4f} acc {val_acc:.4f}')
    
    print("-" * 100)
    print(f"\nTraining complete!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
