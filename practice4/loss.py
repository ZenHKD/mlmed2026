"""
LUNA16 Loss Functions:
  Stage 1: Dice + BCE for lung segmentation
  Stage 2: BCE for candidate classification
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class SegmentationLoss(nn.Module):
    """Combined Dice + BCE loss for lung segmentation."""
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, D, H, W) logits
            target: (B, 1, D, H, W) binary mask
        """
        loss_dice = self.dice(torch.sigmoid(pred), target)
        loss_bce = self.bce(pred, target)
        total = self.dice_weight * loss_dice + self.bce_weight * loss_bce
        return {"total_loss": total, "dice_loss": loss_dice, "bce_loss": loss_bce}


class ClassificationLoss(nn.Module):
    """BCE loss for nodule candidate classification."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1) logits
            target: (B, 1) binary labels
        """
        loss = self.bce(pred, target)
        return {"total_loss": loss}
