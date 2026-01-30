# Phase 2: Train on Infection Segmentation + Fine-tune Classification
# Uses the smaller Infection Segmentation Data (5,826 images)
# Loads Phase 1 model and unfreezes infection decoder

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


class InfectionSegmentationDataset(Dataset):
    """
    Dataset for Infection Segmentation Data.
    
    Structure:
        archive/Infection Segmentation Data/Infection Segmentation Data/
            Train/
                COVID-19/images/, COVID-19/lung masks/, COVID-19/infection masks/
                Non-COVID/images/, Non-COVID/lung masks/  (no infection masks)
                Normal/images/, Normal/lung masks/  (no infection masks)
            Val/
                ...
            Test/
                ...
    
    Note: Only COVID-19 cases have infection masks. For Non-COVID and Normal,
    infection mask is all zeros (no infection).
    """
    
    CLASS_MAP = {'Normal': 0, 'Non-COVID': 1, 'COVID-19': 2}
    
    def __init__(self, data_root, split='Train', img_size=(256, 256)):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        # Collect all samples
        self.samples = []
        
        split_dir = os.path.join(data_root, 'Infection Segmentation Data', 
                                  'Infection Segmentation Data', split)
        
        for class_name in ['COVID-19', 'Non-COVID', 'Normal']:
            class_dir = os.path.join(split_dir, class_name)
            images_dir = os.path.join(class_dir, 'images')
            lung_masks_dir = os.path.join(class_dir, 'lung masks')
            # Infection masks only exist for COVID-19
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
                    infection_mask_path = None  # No infection for Normal/Non-COVID
                
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
        # Count per class
        for cn in ['Normal', 'Non-COVID', 'COVID-19']:
            count = sum(1 for s in self.samples if s['class_name'] == cn)
            print(f"  {cn}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(sample['img_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        
        # Load lung mask
        lung_mask = cv2.imread(sample['lung_mask_path'], cv2.IMREAD_GRAYSCALE)
        lung_mask = cv2.resize(lung_mask, self.img_size)
        lung_mask = (lung_mask > 127).astype(np.float32)
        lung_mask = torch.from_numpy(lung_mask).unsqueeze(0)  # (1, H, W)
        
        # Load infection mask (zeros if no infection)
        if sample['infection_mask_path'] and os.path.exists(sample['infection_mask_path']):
            inf_mask = cv2.imread(sample['infection_mask_path'], cv2.IMREAD_GRAYSCALE)
            inf_mask = cv2.resize(inf_mask, self.img_size)
            inf_mask = (inf_mask > 127).astype(np.float32)
        else:
            # No infection for Normal/Non-COVID cases
            inf_mask = np.zeros(self.img_size, dtype=np.float32)
        inf_mask = torch.from_numpy(inf_mask).unsqueeze(0)  # (1, H, W)
        
        label = sample['label']
        is_covid = 1.0 if sample['class_name'] == 'COVID-19' else 0.0  # Flag for infection loss
        
        return img, lung_mask, inf_mask, label, is_covid


def train_epoch(model, loader, seg_criterion, cls_criterion, optimizer, device, 
                 seg_weight=1.0, cls_weight=0.3):
    """Train for one epoch with lung + infection segmentation + classification."""
    model.train()
    total_loss = 0
    total_lung_loss = 0
    total_inf_loss = 0
    total_cls_loss = 0
    total_lung_dice = 0
    total_inf_dice = 0
    total_inf_count = 0  # Count COVID-19 samples for infection dice
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, lung_masks, inf_masks, labels, is_covid in pbar:
        images = images.to(device)
        lung_masks = lung_masks.to(device)
        inf_masks = inf_masks.to(device)
        labels = labels.to(device)
        is_covid = is_covid.to(device)
        
        optimizer.zero_grad()
        
        # Phase 2: use_infection=True for classification
        lung_pred, inf_pred, class_pred = model(images, use_infection=True)
        
        # Lung segmentation loss (all samples)
        lung_loss = seg_criterion(lung_pred, lung_masks)
        
        # Infection segmentation loss (ONLY COVID-19 cases)
        covid_mask = is_covid.view(-1, 1, 1, 1)  # Shape for broadcasting
        if covid_mask.sum() > 0:
            # Only compute loss where is_covid=1
            inf_pred_covid = inf_pred * covid_mask
            inf_masks_covid = inf_masks * covid_mask
            inf_loss = seg_criterion(inf_pred_covid, inf_masks_covid)
        else:
            inf_loss = torch.tensor(0.0, device=device)
        
        # Classification loss
        cls_loss = cls_criterion(class_pred, labels)
        
        # Combined loss
        loss = seg_weight * lung_loss + seg_weight * inf_loss + cls_weight * cls_loss
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_lung_loss += lung_loss.item()
        total_inf_loss += inf_loss.item()
        total_cls_loss += cls_loss.item()
        total_lung_dice += dice_score(lung_pred, lung_masks)
        
        # Infection dice only for COVID-19 cases
        covid_indices = (is_covid == 1).nonzero(as_tuple=True)[0]
        if len(covid_indices) > 0:
            for idx in covid_indices:
                total_inf_dice += dice_score(inf_pred[idx:idx+1], inf_masks[idx:idx+1])
                total_inf_count += 1
        
        # Classification accuracy
        pred_labels = class_pred.argmax(dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0
    inf_dice = total_inf_dice / total_inf_count if total_inf_count > 0 else 0
    return (total_loss / n, total_lung_loss / n, total_inf_loss / n, 
            total_cls_loss / n, total_lung_dice / n, inf_dice, acc)


def evaluate(model, loader, seg_criterion, cls_criterion, device, 
             seg_weight=1.0, cls_weight=0.3):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_lung_loss = 0
    total_inf_loss = 0
    total_cls_loss = 0
    total_lung_dice = 0
    total_inf_dice = 0
    total_inf_count = 0  # Count COVID-19 samples
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, lung_masks, inf_masks, labels, is_covid in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device)
            lung_masks = lung_masks.to(device)
            inf_masks = inf_masks.to(device)
            labels = labels.to(device)
            is_covid = is_covid.to(device)
            
            lung_pred, inf_pred, class_pred = model(images, use_infection=True)
            
            # Losses
            lung_loss = seg_criterion(lung_pred, lung_masks)
            
            # Infection loss only for COVID-19
            covid_mask = is_covid.view(-1, 1, 1, 1)
            if covid_mask.sum() > 0:
                inf_pred_covid = inf_pred * covid_mask
                inf_masks_covid = inf_masks * covid_mask
                inf_loss = seg_criterion(inf_pred_covid, inf_masks_covid)
            else:
                inf_loss = torch.tensor(0.0, device=device)
            
            cls_loss = cls_criterion(class_pred, labels)
            loss = seg_weight * lung_loss + seg_weight * inf_loss + cls_weight * cls_loss
            
            # Metrics
            total_loss += loss.item()
            total_lung_loss += lung_loss.item()
            total_inf_loss += inf_loss.item()
            total_cls_loss += cls_loss.item()
            total_lung_dice += dice_score(lung_pred, lung_masks)
            
            # Infection dice only for COVID-19
            covid_indices = (is_covid == 1).nonzero(as_tuple=True)[0]
            if len(covid_indices) > 0:
                for idx in covid_indices:
                    total_inf_dice += dice_score(inf_pred[idx:idx+1], inf_masks[idx:idx+1])
                    total_inf_count += 1
            
            # Classification accuracy
            pred_labels = class_pred.argmax(dim=1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += labels.size(0)
    
    n = len(loader)
    acc = total_correct / total_samples if total_samples > 0 else 0
    inf_dice = total_inf_dice / total_inf_count if total_inf_count > 0 else 0
    return (total_loss / n, total_lung_loss / n, total_inf_loss / n, 
            total_cls_loss / n, total_lung_dice / n, inf_dice, acc)


def main():
    # ============== CONFIG ==============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'archive')
    
    # Paths
    phase1_model_path = os.path.join(script_dir, 'phase1_best_model.pth')
    best_model_path = os.path.join(script_dir, 'phase2_best_model.pth')
    last_model_path = os.path.join(script_dir, 'phase2_last_model.pth')
    log_path = os.path.join(script_dir, 'phase2_training_log.csv')
    
    # Hyperparameters
    img_size = (256, 256)
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4  # Increased LR for better infection learning
    seg_weight = 1.0
    cls_weight = 0.3  # Lower cls weight since classification is already trained
    
    print("=" * 80)
    print("Phase 2: Infection Segmentation Training")
    print("=" * 80)
    
    # ============== DATASET ==============
    train_dataset = InfectionSegmentationDataset(data_root, split='Train', img_size=img_size)
    val_dataset = InfectionSegmentationDataset(data_root, split='Val', img_size=img_size)
    
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
    
    # ============== LOAD PHASE 1 CHECKPOINT ==============
    if not os.path.exists(phase1_model_path):
        print(f"ERROR: Phase 1 model not found at {phase1_model_path}")
        print("Please run train1.py first to train Phase 1.")
        return
    
    checkpoint = torch.load(phase1_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded Phase 1 model from epoch {checkpoint['epoch']}")
    print(f"Phase 1 val_dice: {checkpoint['val_dice']:.4f}, val_acc: {checkpoint.get('val_acc', 'N/A')}")
    
    # ============== UNFREEZE INFECTION DECODER ==============
    # In Phase 2, all layers are unfrozen (fine-tune everything)
    for param in model.parameters():
        param.requires_grad = True
    
    print("\nPhase 2: All layers unfrozen for fine-tuning")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============== LOSS & OPTIMIZER ==============
    seg_criterion = BCEDiceLoss(bce_weight=0.5)
    cls_criterion = torch.nn.CrossEntropyLoss()
    
    # Differential learning rate: lower for pretrained, higher for infection decoder
    pretrained_params = list(model.encoder.parameters()) + \
                        list(model.ppm.parameters()) + \
                        list(model.decoder.lung_blocks.parameters()) + \
                        list(model.decoder.lung_cross_attn.parameters()) + \
                        list(model.lung_head.parameters()) + \
                        list(model.classification_head.conv_lung.parameters()) + \
                        list(model.classification_head.fc.parameters())
    
    new_params = list(model.decoder.inf_blocks.parameters()) + \
                 list(model.decoder.inf_cross_attn.parameters()) + \
                 list(model.infection_head.parameters()) + \
                 list(model.classification_head.conv_combined.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': learning_rate * 5}  # 5x higher LR for infection components
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # ============== TRAINING LOOP ==============
    best_val_inf_dice = 0.0
    training_history = []
    
    print("\nStarting training...")
    print("-" * 120)
    
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, seg_criterion, cls_criterion, 
            optimizer, device, seg_weight, cls_weight
        )
        train_loss, train_lung, train_inf, train_cls, train_lung_dice, train_inf_dice, train_acc = train_metrics
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, seg_criterion, cls_criterion, 
            device, seg_weight, cls_weight
        )
        val_loss, val_lung, val_inf, val_cls, val_lung_dice, val_inf_dice, val_acc = val_metrics
        
        # Step scheduler
        scheduler.step()
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_loss,
            'train_lung_loss': train_lung,
            'train_inf_loss': train_inf,
            'train_cls_loss': train_cls,
            'train_lung_dice': train_lung_dice,
            'train_inf_dice': train_inf_dice,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_lung_loss': val_lung,
            'val_inf_loss': val_inf,
            'val_cls_loss': val_cls,
            'val_lung_dice': val_lung_dice,
            'val_inf_dice': val_inf_dice,
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
            'val_lung_dice': val_lung_dice,
            'val_inf_dice': val_inf_dice,
            'val_acc': val_acc,
        }, last_model_path)
        
        # Save best model (based on infection Dice - the new task)
        if val_inf_dice > best_val_inf_dice:
            best_val_inf_dice = val_inf_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_lung_dice': val_lung_dice,
                'val_inf_dice': val_inf_dice,
                'val_acc': val_acc,
            }, best_model_path)
            print(f'Epoch {epoch:03d}  |  lr {current_lr:.2e}  |  '
                  f'train: L {train_lung_dice:.4f} I {train_inf_dice:.4f} acc {train_acc:.4f}  |  '
                  f'val: L {val_lung_dice:.4f} I {val_inf_dice:.4f} acc {val_acc:.4f}  |  *** Best ***')
        else:
            print(f'Epoch {epoch:03d}  |  lr {current_lr:.2e}  |  '
                  f'train: L {train_lung_dice:.4f} I {train_inf_dice:.4f} acc {train_acc:.4f}  |  '
                  f'val: L {val_lung_dice:.4f} I {val_inf_dice:.4f} acc {val_acc:.4f}')
    
    print("-" * 120)
    print(f"\nTraining complete!")
    print(f"Best validation Infection Dice: {best_val_inf_dice:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
