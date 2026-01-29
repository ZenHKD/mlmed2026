# Loss Functions for Multi-Task Segmentation

import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) sigmoid output
            target: (B, 1, H, W) binary mask
        Returns:
            Dice loss (scalar)
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for segmentation.
    Loss = α * BCE + (1-α) * Dice
    """
    
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) sigmoid output
            target: (B, 1, H, W) binary mask
        Returns:
            Combined loss (scalar)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL = -α * (1 - p)^γ * log(p)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) sigmoid output
            target: (B, 1, H, W) binary mask
        Returns:
            Focal loss (scalar)
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for combined segmentation and classification.
    
    Total Loss = λ₁ * L_lung + λ₂ * L_infection + λ₃ * L_classification
    
    Where:
        - L_lung: BCE + Dice for lung segmentation
        - L_infection: BCE + Dice for infection segmentation (only for samples with infection mask)
        - L_classification: CrossEntropy for 3-class classification
    """
    
    def __init__(self, 
                 lung_weight=1.0,
                 infection_weight=1.0, 
                 classification_weight=0.5,
                 bce_weight=0.5,
                 smooth=1.0):
        super().__init__()
        
        self.lung_weight = lung_weight
        self.infection_weight = infection_weight
        self.classification_weight = classification_weight
        
        # Segmentation losses
        self.seg_loss = BCEDiceLoss(bce_weight=bce_weight, smooth=smooth)
        
        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, lung_pred, infection_pred, class_pred,
                lung_target, infection_target, class_target,
                has_infection_mask=None):
        """
        Args:
            lung_pred: (B, 1, H, W) predicted lung mask
            infection_pred: (B, 1, H, W) predicted infection mask
            class_pred: (B, 3) class logits
            lung_target: (B, 1, H, W) ground truth lung mask
            infection_target: (B, 1, H, W) ground truth infection mask
            class_target: (B,) class labels (0=Normal, 1=Non-COVID, 2=COVID)
            has_infection_mask: (B,) bool tensor indicating which samples have valid infection masks
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses for logging
        """
        # Lung segmentation loss (always computed)
        lung_loss = self.seg_loss(lung_pred, lung_target)
        
        # Infection segmentation loss (only for samples with infection mask)
        if has_infection_mask is not None and has_infection_mask.any():
            # Select only samples with infection masks
            mask = has_infection_mask
            infection_loss = self.seg_loss(
                infection_pred[mask], 
                infection_target[mask]
            )
        else:
            # If no infection masks in batch, use all samples
            infection_loss = self.seg_loss(infection_pred, infection_target)
        
        # Classification loss
        cls_loss = self.cls_loss(class_pred, class_target)
        
        # Total loss
        total_loss = (
            self.lung_weight * lung_loss +
            self.infection_weight * infection_loss +
            self.classification_weight * cls_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'lung': lung_loss.item(),
            'infection': infection_loss.item(),
            'classification': cls_loss.item()
        }
        
        return total_loss, loss_dict


class SegmentationLoss(nn.Module):
    """
    Simple segmentation loss for single-task training.
    Uses BCE + Dice combination.
    """
    
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.loss = BCEDiceLoss(bce_weight=bce_weight, smooth=smooth)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


# Metrics (not losses, but useful for evaluation)
def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Calculate Dice score for evaluation.
    
    Args:
        pred: (B, 1, H, W) sigmoid output
        target: (B, 1, H, W) binary mask
        threshold: Threshold to binarize predictions
    Returns:
        Dice score (scalar)
    """
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score.
    
    Args:
        pred: (B, 1, H, W) sigmoid output
        target: (B, 1, H, W) binary mask
        threshold: Threshold to binarize predictions
    Returns:
        IoU score (scalar)
    """
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    # Create dummy data
    B, H, W = 2, 256, 256
    lung_pred = torch.sigmoid(torch.randn(B, 1, H, W))
    lung_target = torch.randint(0, 2, (B, 1, H, W)).float()
    infection_pred = torch.sigmoid(torch.randn(B, 1, H, W))
    infection_target = torch.randint(0, 2, (B, 1, H, W)).float()
    class_pred = torch.randn(B, 3)
    class_target = torch.randint(0, 3, (B,))
    
    # Test individual losses
    dice_loss = DiceLoss()
    bce_dice = BCEDiceLoss()
    focal = FocalLoss()
    
    print(f"  Dice Loss: {dice_loss(lung_pred, lung_target):.4f}")
    print(f"  BCE+Dice Loss: {bce_dice(lung_pred, lung_target):.4f}")
    print(f"  Focal Loss: {focal(lung_pred, lung_target):.4f}")
    
    # Test multi-task loss
    multi_loss = MultiTaskLoss()
    total, loss_dict = multi_loss(
        lung_pred, infection_pred, class_pred,
        lung_target, infection_target, class_target
    )
    
    print(f"\n  Multi-Task Loss:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")
    
    # Test metrics
    print(f"\n  Dice Score: {dice_score(lung_pred, lung_target):.4f}")
    print(f"  IoU Score: {iou_score(lung_pred, lung_target):.4f}")
    
    print("\n  All loss tests passed!")
