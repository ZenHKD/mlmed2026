"""
Visualization of Stage 1 (Lung Segmentation) and Stage 2 (Nodule Detection) predictions.

Usage:
  python practice4/visualize.py

Saves images to practice4/images/
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from practice4.model import UNet3D, ResNet3D18
from practice4.dataloader import load_scan, normalize, resample_isotropic, world_to_voxel
from practice4.evaluate import crop_patch
import scipy.ndimage


def visualize_stage1(args, device):
    """Visualize Stage 1: Lung segmentation predictions on sample scans."""
    print('\n' + '='*60)
    print('  Stage 1: Lung Segmentation Visualization')
    print('='*60)
    
    # Load model
    model_path = os.path.join(args.model_dir, 'stage1', 'fold1', 'best_model.pth')
    model = UNet3D(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded: {model_path}')
    
    # Find some scans from subset 0 (fold 1 val set)
    subset_dir = os.path.join(args.data_dir, 'subset0', 'subset0')
    seg_dir = os.path.join(args.data_dir, 'seg-lungs-LUNA16')
    
    mhd_files = sorted([f for f in os.listdir(subset_dir) if f.endswith('.mhd')])
    selected = mhd_files[:args.num_scans]
    
    save_dir = os.path.join(args.output_dir, 'stage1')
    os.makedirs(save_dir, exist_ok=True)
    
    for scan_file in selected:
        uid = scan_file.replace('.mhd', '')
        print(f'\nProcessing: {uid[:20]}...')
        
        # Load scan
        img_arr, origin, spacing = load_scan(os.path.join(subset_dir, scan_file))
        img_arr = normalize(img_arr)
        img_arr, _ = resample_isotropic(img_arr, spacing, args.target_spacing, order=1)
        
        # Load GT mask
        mask_path = os.path.join(seg_dir, uid + '.mhd')
        if os.path.exists(mask_path):
            import SimpleITK as sitk
            mask_itk = sitk.ReadImage(mask_path)
            gt_mask = sitk.GetArrayFromImage(mask_itk)
            gt_mask, _ = resample_isotropic(gt_mask, spacing, args.target_spacing, order=0)
            gt_mask = (gt_mask > 0).astype(np.float32)
        else:
            gt_mask = None
        
        # Predict in chunks
        D, H, W = img_arr.shape
        pred_mask = np.zeros_like(img_arr, dtype=np.float32)
        crop_size = 64
        
        # Resize H,W to 256 for model input (model was trained with 256x256)
        zoom_y = 256 / H
        zoom_x = 256 / W
        
        for z_start in range(0, D, crop_size):
            z_end = min(z_start + crop_size, D)
            chunk = img_arr[z_start:z_end]
            
            # Pad depth if needed
            actual_d = chunk.shape[0]
            if actual_d < crop_size:
                chunk = np.pad(chunk, ((0, crop_size - actual_d), (0, 0), (0, 0)))
            
            # Resize H,W
            chunk_resized = scipy.ndimage.zoom(chunk, (1.0, zoom_y, zoom_x), order=1)
            
            # Predict
            x = torch.from_numpy(chunk_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                out = model(x)
                out = torch.sigmoid(out).squeeze().cpu().float().numpy()
            
            # Resize back to original H,W
            out = scipy.ndimage.zoom(out.astype(np.float32), (1.0, 1.0/zoom_y, 1.0/zoom_x), order=1)
            pred_mask[z_start:z_start+actual_d] = out[:actual_d, :H, :W]
        
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        # Select representative slices (evenly spaced through the volume)
        n_slices = 6
        slice_indices = np.linspace(D * 0.2, D * 0.8, n_slices, dtype=int)
        
        fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))
        fig.suptitle(f'Stage 1: Lung Segmentation — {uid[:30]}...', fontsize=14, fontweight='bold')
        
        for col, z in enumerate(slice_indices):
            # Row 1: CT + GT overlay
            axes[0, col].imshow(img_arr[z], cmap='gray', vmin=0, vmax=1)
            if gt_mask is not None and z < gt_mask.shape[0]:
                gt_overlay = np.ma.masked_where(gt_mask[z] < 0.5, gt_mask[z])
                axes[0, col].imshow(gt_overlay, cmap='Greens', alpha=0.3, vmin=0, vmax=1)
            axes[0, col].set_title(f'GT — z={z}', fontsize=10)
            axes[0, col].axis('off')
            
            # Row 2: CT + Prediction overlay
            axes[1, col].imshow(img_arr[z], cmap='gray', vmin=0, vmax=1)
            pred_overlay = np.ma.masked_where(pred_binary[z] < 0.5, pred_binary[z])
            axes[1, col].imshow(pred_overlay, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
            axes[1, col].set_title(f'Pred — z={z}', fontsize=10)
            axes[1, col].axis('off')
        
        # Add legend
        gt_patch = mpatches.Patch(color='green', alpha=0.4, label='GT Lung')
        pred_patch = mpatches.Patch(color='blue', alpha=0.4, label='Predicted Lung')
        fig.legend(handles=[gt_patch, pred_patch], loc='lower center', ncol=2, fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f'scan{selected.index(scan_file)+1}_{uid[-15:]}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {save_path}')


def visualize_stage2(args, device):
    """Visualize Stage 2: Nodule candidate predictions on sample scans."""
    print('\n' + '='*60)
    print('  Stage 2: Nodule Detection Visualization')
    print('='*60)
    
    # Load model
    model_path = os.path.join(args.model_dir, 'stage2', 'fold1', 'best_model.pth')
    model = ResNet3D18(in_ch=1, num_classes=1).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded: {model_path}')
    
    # Load annotations and candidates
    annotations = pd.read_csv(args.annotations_file)
    candidates = pd.read_csv(args.candidates_file)
    
    # Find scans from subset 0 that have nodules
    subset_dir = os.path.join(args.data_dir, 'subset0', 'subset0')
    available_uids = set()
    uid_to_path = {}
    for f in sorted(os.listdir(subset_dir)):
        if f.endswith('.mhd'):
            uid = f.replace('.mhd', '')
            available_uids.add(uid)
            uid_to_path[uid] = os.path.join(subset_dir, f)
    
    # Filter to scans with nodules
    scans_with_nodules = annotations[annotations['seriesuid'].isin(available_uids)]
    selected_uids = scans_with_nodules['seriesuid'].unique()[:args.num_scans]
    
    save_dir = os.path.join(args.output_dir, 'stage2')
    os.makedirs(save_dir, exist_ok=True)
    
    target_sp = np.array([args.target_spacing] * 3)
    
    for uid in selected_uids:
        print(f'\nProcessing: {uid[:20]}...')
        
        # Load and resample scan
        img_arr, origin, spacing = load_scan(uid_to_path[uid])
        img_arr = normalize(img_arr)
        img_arr, _ = resample_isotropic(img_arr, spacing, args.target_spacing, order=1)
        
        # Get GT nodules
        scan_annots = annotations[annotations['seriesuid'] == uid]
        gt_nodules = []
        for _, row in scan_annots.iterrows():
            center_world = np.array([row['coordZ'], row['coordY'], row['coordX']])
            center_vox = world_to_voxel(center_world, origin, target_sp)
            radius_vox = (row['diameter_mm'] / 2.0) / args.target_spacing
            gt_nodules.append({
                'center_vox': center_vox,
                'radius_vox': radius_vox,
                'diameter_mm': row['diameter_mm']
            })
        
        # Get candidates for this scan and predict
        scan_cands = candidates[candidates['seriesuid'] == uid]
        pred_nodules = []
        pred_non_nodules = []
        
        for _, row in scan_cands.iterrows():
            center_world = np.array([row['coordZ'], row['coordY'], row['coordX']])
            center_vox = world_to_voxel(center_world, origin, target_sp)
            
            patch = crop_patch(img_arr, center_vox, args.patch_size)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logit = model(patch_tensor)
                prob = torch.sigmoid(logit).item()
            
            info = {'center_vox': center_vox, 'prob': prob, 'gt_class': row['class']}
            if prob > 0.5:
                pred_nodules.append(info)
            else:
                pred_non_nodules.append(info)
        
        print(f'  GT nodules: {len(gt_nodules)}, Predicted nodules: {len(pred_nodules)}, '
              f'Rejected: {len(pred_non_nodules)}')
        
        # Visualize slices at each GT nodule location
        for nod_idx, nodule in enumerate(gt_nodules):
            z_center = int(round(nodule['center_vox'][0]))
            z_center = np.clip(z_center, 0, img_arr.shape[0] - 1)
            
            # Show 3 consecutive slices around the nodule
            slices = [z_center - 1, z_center, z_center + 1]
            slices = [s for s in slices if 0 <= s < img_arr.shape[0]]
            
            fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
            if len(slices) == 1:
                axes = [axes]
            
            fig.suptitle(
                f'Stage 2: Nodule Detection — {uid[:25]}...\n'
                f'GT Nodule {nod_idx+1}: diameter={nodule["diameter_mm"]:.1f}mm',
                fontsize=13, fontweight='bold'
            )
            
            for col, z in enumerate(slices):
                ax = axes[col]
                ax.imshow(img_arr[z], cmap='gray', vmin=0, vmax=1)
                
                # Draw GT nodule circle (green)
                nod_y, nod_x = nodule['center_vox'][1], nodule['center_vox'][2]
                r = nodule['radius_vox']
                if abs(z - nodule['center_vox'][0]) <= r:
                    # Circle radius at this slice
                    dz = abs(z - nodule['center_vox'][0])
                    slice_r = np.sqrt(max(r**2 - dz**2, 0))
                    circle_gt = plt.Circle((nod_x, nod_y), slice_r, 
                                           color='lime', fill=False, linewidth=2, linestyle='-')
                    ax.add_patch(circle_gt)
                
                # Draw predicted detections near this slice (red = TP, orange = FP)
                for pred in pred_nodules:
                    pz, py, px = pred['center_vox']
                    if abs(z - pz) < 3:  # within 3 slices
                        color = 'red' if pred['gt_class'] == 1 else 'orange'
                        marker_size = max(4, pred['prob'] * 12)
                        ax.plot(px, py, 'o', color=color, markersize=marker_size, 
                                markeredgecolor='white', markeredgewidth=0.5, alpha=0.8)
                        ax.annotate(f'{pred["prob"]:.2f}', (px + 5, py - 5),
                                    color=color, fontsize=8, fontweight='bold')
                
                is_center = "← center" if z == z_center else ""
                ax.set_title(f'z={z} {is_center}', fontsize=11)
                ax.axis('off')
            
            # Legend
            gt_circ = mpatches.Patch(edgecolor='lime', facecolor='none', linewidth=2, label='GT Nodule')
            tp_dot = mpatches.Patch(color='red', label='Pred: True Positive')
            fp_dot = mpatches.Patch(color='orange', label='Pred: False Positive')
            fig.legend(handles=[gt_circ, tp_dot, fp_dot], loc='lower center', ncol=3, fontsize=11)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.9])
            save_path = os.path.join(save_dir, f'scan{list(selected_uids).index(uid)+1}_{uid[-15:]}_nodule{nod_idx+1}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {save_path}')
        
        # Also save an overview with top predictions across the scan
        top_preds = sorted(pred_nodules, key=lambda x: -x['prob'])[:12]
        if top_preds:
            # Get unique z-slices of top predictions
            z_slices = sorted(set(int(round(p['center_vox'][0])) for p in top_preds))[:6]
            
            fig, axes = plt.subplots(1, len(z_slices), figsize=(5 * len(z_slices), 5))
            if len(z_slices) == 1:
                axes = [axes]
            
            fig.suptitle(f'Stage 2: Top Predictions Overview — {uid[:25]}...', 
                         fontsize=13, fontweight='bold')
            
            for col, z in enumerate(z_slices):
                ax = axes[col]
                z = np.clip(z, 0, img_arr.shape[0] - 1)
                ax.imshow(img_arr[z], cmap='gray', vmin=0, vmax=1)
                
                # Draw GT
                for nodule in gt_nodules:
                    nod_y, nod_x = nodule['center_vox'][1], nodule['center_vox'][2]
                    r = nodule['radius_vox']
                    dz = abs(z - nodule['center_vox'][0])
                    if dz <= r:
                        slice_r = np.sqrt(max(r**2 - dz**2, 0))
                        circle = plt.Circle((nod_x, nod_y), slice_r,
                                           color='lime', fill=False, linewidth=2)
                        ax.add_patch(circle)
                
                # Draw predictions near this slice
                for pred in top_preds:
                    pz, py, px = pred['center_vox']
                    if abs(z - pz) < 2:
                        color = 'red' if pred['gt_class'] == 1 else 'orange'
                        ax.plot(px, py, 'o', color=color, markersize=8,
                                markeredgecolor='white', markeredgewidth=0.5)
                        ax.annotate(f'{pred["prob"]:.2f}', (px + 5, py - 5),
                                    color=color, fontsize=8, fontweight='bold')
                
                ax.set_title(f'z={z}', fontsize=11)
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            save_path = os.path.join(save_dir, f'scan{list(selected_uids).index(uid)+1}_{uid[-15:]}_overview.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize LUNA16 predictions')
    parser.add_argument('--model_dir', type=str, default='practice4/weights')
    parser.add_argument('--data_dir', type=str, default='practice4/data')
    parser.add_argument('--annotations_file', type=str,
                        default='practice4/data/annotations.csv')
    parser.add_argument('--candidates_file', type=str,
                        default='practice4/data/candidates_V2/candidates_V2.csv')
    parser.add_argument('--output_dir', type=str, default='practice4/images')
    parser.add_argument('--num_scans', type=int, default=3,
                        help='Number of scans to visualize')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--target_spacing', type=float, default=1.0)
    parser.add_argument('--stage', type=int, default=0,
                        help='1=seg only, 2=det only, 0=both (default)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stage in [0, 1]:
        visualize_stage1(args, device)
    
    if args.stage in [0, 2]:
        visualize_stage2(args, device)
    
    print(f'\nAll images saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
