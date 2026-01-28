import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import pandas as pd
from model import CU_Net
from extract_label import label


class HCDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=(256, 256)):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, filename)
        
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        
        # Load annotation mask
        annot_name = filename.replace('.png', '_Annotation.png')
        annot_path = os.path.join(self.img_dir, annot_name)
        mask = label(annot_path)  # Returns (H, W, 1)
        mask = cv2.resize(mask.squeeze(), self.img_size)
        mask = torch.tensor(mask).unsqueeze(0).float()  # (1, H, W)
        
        return img, mask


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient (similar to F1 for segmentation)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice += dice_score(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            total_loss += criterion(outputs, masks).item()
            total_dice += dice_score(outputs, masks).item()
            total_iou += iou_score(outputs, masks).item()
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, 'data/training_set_pixel_size_and_HC.csv')
    train_dir = os.path.join(script_dir, 'data/training_set')
    best_model_path = os.path.join(script_dir, 'best_model.pth')
    log_path = os.path.join(script_dir, 'training_log.csv')
    
    # Hyperparameters
    img_size = (256, 256)
    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-4
    val_split = 0.1
    
    # Dataset
    full_dataset = HCDataset(train_csv, train_dir, img_size=img_size)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split
    split_path = os.path.join(script_dir, 'data_split.csv')
    
    if os.path.exists(split_path):
        df_split = pd.read_csv(split_path)
        train_filenames = df_split['train'].dropna().tolist()
        val_filenames = df_split['val'].dropna().tolist()
        
        # Map filenames to indices
        filename_to_idx = {name: idx for idx, name in enumerate(full_dataset.df['filename'])}
        
        train_indices = [filename_to_idx[name] for name in train_filenames if name in filename_to_idx]
        val_indices = [filename_to_idx[name] for name in val_filenames if name in filename_to_idx]
        
        print(f"Loaded existing split from {split_path}")
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
        # Fixed random shuffle for reproducibility
        g = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)
        
        # Get filenames from indices
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        
        train_filenames = full_dataset.df.iloc[train_indices]['filename'].values
        val_filenames = full_dataset.df.iloc[val_indices]['filename'].values
        
        # Save filenames
        df_split = pd.DataFrame({
            'train': pd.Series(train_filenames),
            'val': pd.Series(val_filenames)
        })
        df_split.to_csv(split_path, index=False)
        print(f"Created and saved new split to {split_path}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size} - ({val_split*100}%)")
    print(f"Image size: {img_size}, Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 100)
    
    # Model
    model = CU_Net(in_channels=1, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_val_dice = 0.0
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_dice)
            
        # Log metrics
        epoch_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou
        }
        training_history.append(epoch_log)
        
        # Save training history to CSV immediately
        pd.DataFrame(training_history).to_csv(log_path, index=False)
            
        # Save best model (based on val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f'Epoch {epoch:03d}  ||  train_loss {train_loss:.6f}  train_dice {train_dice:.6f}  train_iou {train_iou:.6f}  ||  '
                  f'val_loss {val_loss:.6f}  val_dice {val_dice:.6f}  val_iou {val_iou:.6f}  ||  *** Best model saved! ***')
        else:
            print(f'Epoch {epoch:03d}  ||  train_loss {train_loss:.6f}  train_dice {train_dice:.6f}  train_iou {train_iou:.6f}  ||  '
                  f'val_loss {val_loss:.6f}  val_dice {val_dice:.6f}  val_iou {val_iou:.6f}')
    
    print("-" * 100)
    print(f"Training complete! Best validation Dice: {best_val_dice:.6f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
