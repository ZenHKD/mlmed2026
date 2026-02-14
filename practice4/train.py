"""
LUNA16 Two-Stage Training Script

Usage:
  Stage 1 (Lung Segmentation):
    python practice4/train.py --stage 1 --epochs 10

  Stage 2 (Candidate Classification):
    python practice4/train.py --stage 2 --epochs 10 --batch_size 32
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import os
import sys
import csv
from tqdm import tqdm
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from practice4.model import UNet3D, ResNet3D18
from practice4.loss import SegmentationLoss, ClassificationLoss
from practice4.dataloader import get_seg_fold_loaders, get_candidate_fold_loaders
from practice4.val import validate_seg, validate_cls


# ============================================================
#  Training Loops
# ============================================================

def train_seg_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch):
    """Train one epoch for lung segmentation."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        image = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            pred = model(image)
            loss_dict = criterion(pred, mask)
            loss = loss_dict['total_loss']
        
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(len(dataloader), 1)


def train_cls_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch):
    """Train one epoch for candidate classification."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        patch = batch['patch'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            pred = model(patch)
            loss_dict = criterion(pred, label)
            loss = loss_dict['total_loss']
        
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Track accuracy
        pred_labels = (torch.sigmoid(pred) > 0.5).float()
        correct += (pred_labels == label).sum().item()
        total += label.numel()
        
        acc = correct / total if total > 0 else 0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.3f}'})
    
    return total_loss / max(len(dataloader), 1)


# ============================================================
#  Fold Training
# ============================================================

def train_seg_fold(fold, args, device):
    """Train Stage 1 (segmentation) for one fold."""
    fold_dir = Path(args.save_dir) / 'stage1' / f'fold{fold + 1}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n{"#"*60}')
    print(f'  STAGE 1 — FOLD {fold + 1}/5 — Lung Segmentation')
    print(f'{"#"*60}')
    
    train_loader, val_loader, train_subsets, val_subsets = get_seg_fold_loaders(
        data_dir=args.data_dir, fold=fold, total_folds=5,
        crop_size=args.crop_size, num_workers=args.num_workers,
        batch_size=1, target_spacing=args.target_spacing
    )
    
    # Info file
    with open(fold_dir / 'info.txt', 'w') as f:
        f.write(f'Stage 1: Lung Segmentation\n')
        f.write(f'Fold {fold+1}/5\n')
        f.write(f'Train subsets: {train_subsets} ({len(train_loader.dataset)} chunks)\n')
        f.write(f'Val subsets:   {val_subsets} ({len(val_loader.dataset)} chunks)\n')
        f.write(f'Epochs: {args.epochs}, LR: {args.lr}\n')
        f.write(f'Crop size: {args.crop_size}, Spacing: {args.target_spacing}mm\n')
    
    model = UNet3D(in_ch=1, out_ch=1).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1000
    )
    
    # CSV log
    log_file = fold_dir / 'training_log.csv'
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'lung_Dice', 'lung_IoU'])
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_seg_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch + 1)
        val_loss, metrics = validate_seg(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        
        print(f'Fold {fold+1} | Ep {epoch+1:3d} | LR {lr:.6f} | '
              f'Train {train_loss:.4f} | Val {val_loss:.4f} | '
              f'Dice {metrics["lung_Dice"]:.4f} | IoU {metrics["lung_IoU"]:.4f}')
        
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, f'{lr:.6f}', f'{train_loss:.4f}', f'{val_loss:.4f}',
                f'{metrics["lung_Dice"]:.4f}', f'{metrics["lung_IoU"]:.4f}'
            ])
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_loss': best_loss, 'val_metrics': metrics,
                'fold': fold, 'train_subsets': train_subsets, 'val_subsets': val_subsets
            }, fold_dir / 'best_model.pth')
            print(f'   Saved best model (loss={best_loss:.4f})')
    
    print(f'Fold {fold+1} done. Best loss: {best_loss:.4f}')
    return best_loss


def train_cls_fold(fold, args, device):
    """Train Stage 2 (classification) for one fold."""
    fold_dir = Path(args.save_dir) / 'stage2' / f'fold{fold + 1}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n{"#"*60}')
    print(f'  STAGE 2 — FOLD {fold + 1}/5 — Candidate Classification')
    print(f'{"#"*60}')
    
    train_dataset, val_loader, train_subsets, val_subsets, resample_train_loader = get_candidate_fold_loaders(
        data_dir=args.data_dir,
        candidates_file=args.candidates_file,
        fold=fold, total_folds=5,
        patch_size=args.patch_size, num_workers=args.num_workers,
        batch_size=args.batch_size, target_spacing=args.target_spacing
    )
    
    # Get first loader to determine steps_per_epoch
    first_loader = resample_train_loader(0)
    steps_per_epoch = len(first_loader)
    
    # Compute stats for info file (same as dataloader prints)
    labels = train_dataset.candidates['class'].values
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    n_scans_train = train_dataset.candidates['seriesuid'].nunique()
    
    val_ds = val_loader.dataset
    # Handle Subset wrapping
    if hasattr(val_ds, 'dataset'):
        val_cands = val_ds.dataset.candidates
        val_n_scans = val_cands['seriesuid'].nunique()
    else:
        val_cands = val_ds.candidates
        val_n_scans = val_cands['seriesuid'].nunique()
    val_n_pos = int((val_cands['class'] == 1).sum())
    val_n_neg = int((val_cands['class'] == 0).sum())
    
    # Info file
    with open(fold_dir / 'info.txt', 'w') as f:
        f.write(f'Stage 2: Candidate Classification\n')
        f.write(f'Fold {fold+1}/5\n')
        f.write(f'Train: subsets {train_subsets}\n')
        f.write(f'Val:   subsets {val_subsets}\n')
        f.write(f'CandidateDataset: {len(train_dataset)} candidates ({n_pos} positive, {n_neg} negative) from {n_scans_train} scans\n')
        f.write(f'CandidateDataset: {len(val_cands)} candidates ({val_n_pos} positive, {val_n_neg} negative) from {val_n_scans} scans\n')
        f.write(f'Training: {n_pos} pos, {n_neg} neg total\n')
        f.write(f'Per epoch: {n_pos} pos + {n_pos} neg = {n_pos * 2} samples\n')
        f.write(f'~{n_neg // max(n_pos, 1)} epochs to see all negatives\n')
        f.write(f'Validation: {len(val_loader.dataset)} samples\n')
    
    model = ResNet3D18(in_ch=1, num_classes=1).to(device)
    criterion = ClassificationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1000
    )
    
    # CSV log
    log_file = fold_dir / 'training_log.csv'
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'lr', 'train_loss', 'val_loss',
            'accuracy', 'precision', 'recall', 'f1', 'auc_roc'
        ])
    
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        # Resample negatives each epoch so model sees different negatives
        train_loader = resample_train_loader(epoch)
        
        train_loss = train_cls_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch + 1)
        val_loss, metrics = validate_cls(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        
        print(f'Fold {fold+1} | Ep {epoch+1:3d} | LR {lr:.6f} | '
              f'Train {train_loss:.4f} | Val {val_loss:.4f} | '
              f'Acc {metrics["accuracy"]:.4f} | F1 {metrics["f1"]:.4f} | '
              f'AUC {metrics["auc_roc"]:.4f} | '
              f'(+{metrics["n_pos"]} / -{metrics["n_neg"]})')
        
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, f'{lr:.6f}', f'{train_loss:.4f}', f'{val_loss:.4f}',
                f'{metrics["accuracy"]:.4f}', f'{metrics["precision"]:.4f}',
                f'{metrics["recall"]:.4f}', f'{metrics["f1"]:.4f}',
                f'{metrics["auc_roc"]:.4f}'
            ])
        
        # Save best by AUC-ROC (better than loss for imbalanced data)
        if metrics['auc_roc'] > best_auc:
            best_auc = metrics['auc_roc']
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_auc': best_auc, 'val_metrics': metrics,
                'fold': fold, 'train_subsets': train_subsets, 'val_subsets': val_subsets
            }, fold_dir / 'best_model.pth')
            print(f'   Saved best model (AUC={best_auc:.4f})')
    
    print(f'Fold {fold+1} done. Best AUC: {best_auc:.4f}')
    return best_auc


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LUNA16 Two-Stage Training')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='1=segmentation, 2=classification')
    parser.add_argument('--data_dir', type=str, default='practice4/data')
    parser.add_argument('--candidates_file', type=str, 
                        default='practice4/data/candidates_V2/candidates_V2.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (Stage 2 only, Stage 1 always uses 1)')
    parser.add_argument('--crop_size', type=int, default=64,
                        help='Chunk depth for Stage 1')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size for Stage 2')
    parser.add_argument('--target_spacing', type=float, default=1.0,
                        help='Isotropic resampling spacing in mm')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='practice4/weights')
    parser.add_argument('--fold', type=int, default=-1,
                        help='Train specific fold (0-4), or -1 for all folds')
    args = parser.parse_args()
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print('TF32 + cuDNN benchmark enabled')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Stage: {args.stage}')
    
    # Select folds
    folds = [args.fold] if args.fold >= 0 else list(range(5))
    
    results = {}
    for fold in folds:
        if args.stage == 1:
            result = train_seg_fold(fold, args, device)
        else:
            result = train_cls_fold(fold, args, device)
        results[fold + 1] = result
    
    # Summary
    metric_name = "Val Loss" if args.stage == 1 else "AUC-ROC"
    print(f'\n{"#"*60}')
    print(f'  STAGE {args.stage} COMPLETE')
    print(f'{"#"*60}')
    for fold_num, val in results.items():
        print(f'  Fold {fold_num}: Best {metric_name} = {val:.4f}')
    if len(results) > 1:
        avg = sum(results.values()) / len(results)
        print(f'  Average: {avg:.4f}')
    print(f'{"#"*60}\n')


if __name__ == '__main__':
    main()
