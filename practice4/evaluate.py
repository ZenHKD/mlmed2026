"""
LUNA16 FROC Evaluation

Evaluates Stage 2 (candidate classifier) predictions against
ground truth annotations (annotations.csv).

Usage:
  python practice4/evaluate.py

FROC score = average sensitivity at [1/8, 1/4, 1/2, 1, 2, 4, 8] FP/scan
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from practice4.model import ResNet3D18
from practice4.dataloader import load_scan, normalize, resample_isotropic, world_to_voxel


def crop_patch(volume, center_vox, patch_size=32):
    """Crop a 3D patch centered on voxel coordinates."""
    half = patch_size // 2
    D, H, W = volume.shape
    
    cz, cy, cx = int(round(center_vox[0])), int(round(center_vox[1])), int(round(center_vox[2]))
    
    z1, z2 = cz - half, cz + half
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    
    sz1, sz2 = max(0, z1), min(D, z2)
    sy1, sy2 = max(0, y1), min(H, y2)
    sx1, sx2 = max(0, x1), min(W, x2)
    
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    
    pz1 = sz1 - z1
    py1 = sy1 - y1
    px1 = sx1 - x1
    
    patch[pz1:pz1+(sz2-sz1), py1:py1+(sy2-sy1), px1:px1+(sx2-sx1)] = \
        volume[sz1:sz2, sy1:sy2, sx1:sx2]
    
    return patch


def compute_froc(tp_probs, fp_probs, num_nodules, num_scans,
                 fp_rates=[0.125, 0.25, 0.5, 1, 2, 4, 8]):
    """Compute FROC curve and score.
    
    Args:
        tp_probs: list of probabilities for true positive detections
        fp_probs: list of probabilities for false positive detections
        num_nodules: total number of ground truth nodules
        num_scans: total number of scans evaluated
        fp_rates: FP/scan rates at which to compute sensitivity
    
    Returns:
        froc_score, sensitivities_at_fp_rates
    """
    # Sort all predictions by probability (descending)
    all_probs = np.array(tp_probs + fp_probs)
    all_is_tp = np.array([1] * len(tp_probs) + [0] * len(fp_probs))
    
    if len(all_probs) == 0:
        return 0.0, [0.0] * len(fp_rates)
    
    sort_idx = np.argsort(-all_probs)
    all_probs = all_probs[sort_idx]
    all_is_tp = all_is_tp[sort_idx]
    
    sensitivities = []
    for target_fp_rate in fp_rates:
        max_fps = int(target_fp_rate * num_scans)
        
        # Count TPs and FPs as we lower threshold
        tp_count = 0
        fp_count = 0
        
        for i in range(len(all_probs)):
            if all_is_tp[i] == 1:
                tp_count += 1
            else:
                fp_count += 1
            
            if fp_count > max_fps:
                break
        
        sensitivity = tp_count / max(num_nodules, 1)
        sensitivities.append(sensitivity)
    
    froc_score = np.mean(sensitivities)
    return froc_score, sensitivities


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Load model
    model = ResNet3D18(in_ch=1, num_classes=1).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model from {args.model}')
    if 'val_metrics' in ckpt:
        print(f'  Val metrics: {ckpt["val_metrics"]}')
    
    # Load annotations (ground truth)
    annotations = pd.read_csv(args.annotations_file)
    print(f'Annotations: {len(annotations)} nodules in {annotations.seriesuid.nunique()} scans')
    
    # Load candidates
    candidates = pd.read_csv(args.candidates_file)
    
    # Get scans to evaluate (from val subsets or all)
    if args.val_subsets:
        val_subsets = [int(x) for x in args.val_subsets.split(',')]
    else:
        val_subsets = list(range(5))
    
    # Find scan files
    scan_paths = {}
    for subset_idx in val_subsets:
        subset_dir = os.path.join(args.data_dir, f'subset{subset_idx}', f'subset{subset_idx}')
        if not os.path.isdir(subset_dir):
            continue
        for f in os.listdir(subset_dir):
            if f.endswith('.mhd'):
                uid = f.replace('.mhd', '')
                scan_paths[uid] = os.path.join(subset_dir, f)
    
    print(f'Evaluating {len(scan_paths)} scans from subsets {val_subsets}')
    
    # Filter candidates and annotations to our scans
    available_uids = set(scan_paths.keys())
    eval_candidates = candidates[candidates['seriesuid'].isin(available_uids)]
    eval_annotations = annotations[annotations['seriesuid'].isin(available_uids)]
    
    print(f'Candidates to classify: {len(eval_candidates)}')
    print(f'Ground truth nodules: {len(eval_annotations)}')
    
    # Group candidates by scan for efficient loading
    grouped = eval_candidates.groupby('seriesuid')
    
    # Run predictions
    tp_probs = []  # probabilities assigned to true positives
    fp_probs = []  # probabilities assigned to false positives
    all_detections = []  # (uid, x, y, z, prob) for all candidates with prob > threshold
    
    target_sp = np.array([args.target_spacing] * 3)
    
    for uid, group in tqdm(grouped, desc='Evaluating', total=len(grouped)):
        # Load and resample scan
        path = scan_paths[uid]
        img_arr, origin, spacing = load_scan(path)
        img_arr = normalize(img_arr)
        img_arr, _ = resample_isotropic(img_arr, spacing, args.target_spacing, order=1)
        
        # Get ground truth nodules for this scan
        scan_annotations = eval_annotations[eval_annotations['seriesuid'] == uid]
        gt_nodules = []
        for _, row in scan_annotations.iterrows():
            gt_nodules.append({
                'center': np.array([row['coordZ'], row['coordY'], row['coordX']]),
                'radius': row['diameter_mm'] / 2.0  # standard LUNA16 matching
            })
        
        # Predict on each candidate
        for _, row in group.iterrows():
            center_world = np.array([row['coordZ'], row['coordY'], row['coordX']])
            center_vox = world_to_voxel(center_world, origin, target_sp)
            
            # Crop, predict
            patch = crop_patch(img_arr, center_vox, args.patch_size)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logit = model(patch_tensor)
                prob = torch.sigmoid(logit).item()
            
            # Check if this candidate matches any GT nodule (within diameter/2)
            is_tp = False
            for nodule in gt_nodules:
                dist = np.linalg.norm(center_world - nodule['center'])
                if dist < nodule['radius']:
                    is_tp = True
                    break
            
            if is_tp:
                tp_probs.append(prob)
            else:
                fp_probs.append(prob)
            
            if prob > args.det_threshold:
                all_detections.append({
                    'seriesuid': uid,
                    'coordX': row['coordX'],
                    'coordY': row['coordY'],
                    'coordZ': row['coordZ'],
                    'probability': prob
                })
    
    # Compute FROC
    num_scans = len(scan_paths)
    num_nodules = len(eval_annotations)
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    
    froc_score, sensitivities = compute_froc(
        tp_probs, fp_probs, num_nodules, num_scans, fp_rates
    )
    
    # Print results
    print(f'\n{"="*60}')
    print(f'  FROC EVALUATION RESULTS')
    print(f'{"="*60}')
    print(f'Scans evaluated:    {num_scans}')
    print(f'Ground truth nodules: {num_nodules}')
    print(f'True positives:     {len(tp_probs)}')
    print(f'False positives:    {len(fp_probs)}')
    print(f'\nSensitivity at FP/scan rates:')
    for fp_rate, sens in zip(fp_rates, sensitivities):
        print(f'  {fp_rate:>5.3f} FP/scan: {sens:.4f}')
    print(f'\n  FROC Score: {froc_score:.4f}')
    print(f'{"="*60}')
    
    # Save detections
    if args.output:
        det_df = pd.DataFrame(all_detections)
        det_df.to_csv(args.output, index=False)
        print(f'\nDetections saved to {args.output}')
    
    return froc_score


def main():
    parser = argparse.ArgumentParser(description='LUNA16 FROC Evaluation')
    parser.add_argument('--model_dir', type=str, default='practice4/weights/stage2',
                        help='Directory containing fold1-5 subdirectories')
    parser.add_argument('--data_dir', type=str, default='practice4/data')
    parser.add_argument('--annotations_file', type=str,
                        default='practice4/data/annotations.csv')
    parser.add_argument('--candidates_file', type=str,
                        default='practice4/data/candidates_V2/candidates_V2.csv')
    parser.add_argument('--folds', type=str, default='1,2,3,4,5',
                        help='Comma-separated fold numbers to evaluate (default: 1,2,3,4,5)')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--target_spacing', type=float, default=1.0)
    parser.add_argument('--match_radius_mm', type=float, default=5.0,
                        help='Distance threshold for matching predictions to GT (mm)')
    parser.add_argument('--det_threshold', type=float, default=0.5,
                        help='Probability threshold for saving detections')
    args = parser.parse_args()
    
    folds = [int(x) for x in args.folds.split(',')]
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    
    all_results = {}
    
    for fold in folds:
        val_subset = fold - 1  # fold 1 -> subset 0, fold 2 -> subset 1, etc.
        model_path = os.path.join(args.model_dir, f'fold{fold}', 'best_model.pth')
        output_path = os.path.join(args.model_dir, f'fold{fold}', 'detections.csv')
        
        if not os.path.exists(model_path):
            print(f'\n⚠ Fold {fold}: {model_path} not found, skipping')
            continue
        
        print(f'\n{"#"*60}')
        print(f'  FOLD {fold}/5 — Evaluating on subset {val_subset}')
        print(f'{"#"*60}')
        
        eval_args = argparse.Namespace(
            model=model_path,
            data_dir=args.data_dir,
            annotations_file=args.annotations_file,
            candidates_file=args.candidates_file,
            val_subsets=str(val_subset),
            patch_size=args.patch_size,
            target_spacing=args.target_spacing,
            match_radius_mm=args.match_radius_mm,
            det_threshold=args.det_threshold,
            output=output_path
        )
        
        froc_score = evaluate(eval_args)
        all_results[fold] = froc_score
    
    # Print summary
    print(f'\n{"="*60}')
    print(f'  OVERALL SUMMARY — ALL FOLDS')
    print(f'{"="*60}')
    for fold, score in all_results.items():
        print(f'  Fold {fold}: FROC = {score:.4f}')
    if all_results:
        avg = np.mean(list(all_results.values()))
        print(f'\n  Average FROC: {avg:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

