"""
LUNA16 Data Loaders:
  Stage 1: LungSegDataset — Full CT slices for lung segmentation
  Stage 2: CandidateDataset — 32^3 patches for nodule classification
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import scipy.ndimage
from collections import OrderedDict


# ============================================================
#  Shared Utilities
# ============================================================

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    """Normalize CT HU values to [0, 1]."""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return np.clip(image, 0.0, 1.0)

def load_scan(path):
    """Load a CT scan from .mhd file. Returns (array, origin, spacing) in Z,Y,X order."""
    itk_img = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(itk_img)   # (Z, Y, X)
    origin = np.array(list(reversed(itk_img.GetOrigin())))   # Z, Y, X
    spacing = np.array(list(reversed(itk_img.GetSpacing())))  # Z, Y, X
    return img_array, origin, spacing

def resample_isotropic(image, spacing, target_spacing=1.0, order=1):
    """Resample a 3D volume to isotropic spacing.
    
    Args:
        image: (D, H, W) numpy array
        spacing: (3,) current spacing in mm [z, y, x]
        target_spacing: target spacing in mm (scalar or 3-tuple)
        order: interpolation order (1=linear, 0=nearest)
    
    Returns:
        resampled_image, new_spacing
    """
    if isinstance(target_spacing, (int, float)):
        target_spacing = np.array([target_spacing] * 3)
    
    zoom_factors = spacing / target_spacing
    resampled = scipy.ndimage.zoom(image, zoom_factors, order=order)
    return resampled, target_spacing

def world_to_voxel(world_coord, origin, spacing):
    """Convert world coordinates (mm) to voxel coordinates."""
    voxel_coord = np.absolute(world_coord - origin) / spacing
    return voxel_coord


# ============================================================
#  Scan Cache (shared between datasets)
# ============================================================

class ScanCache:
    """LRU cache for loaded + resampled scans to avoid redundant disk I/O."""
    
    def __init__(self, max_size=8):
        self._cache = OrderedDict()
        self._max_size = max_size
    
    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ============================================================
#  Stage 1: Lung Segmentation Dataset
# ============================================================

class LungSegDataset(Dataset):
    """Dataset for lung segmentation training.
    
    Loads full CT scans, resamples to isotropic spacing,
    and returns sliding-window chunks along Z axis.
    """
    
    def __init__(self, data_dir, subset_indices, seg_dir, crop_size=64, target_spacing=1.0, is_train=True):
        super().__init__()
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.is_train = is_train
        self._cache = ScanCache(max_size=4)
        
        # Collect all .mhd files from specified subsets
        self.series_infos = []
        for subset_idx in subset_indices:
            subset_dir = os.path.join(data_dir, f"subset{subset_idx}", f"subset{subset_idx}")
            if not os.path.isdir(subset_dir):
                continue
            for f in sorted(os.listdir(subset_dir)):
                if f.endswith(".mhd"):
                    uid = f.replace(".mhd", "")
                    self.series_infos.append({
                        "uid": uid,
                        "path": os.path.join(subset_dir, f)
                    })
        
        # Build chunk index
        self.chunks = []
        print(f"Indexing chunks for {len(self.series_infos)} scans (crop_size={crop_size})...")
        for i, info in enumerate(self.series_infos):
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(info['path'])
                reader.LoadPrivateTagsOn()
                reader.ReadImageInformation()
                size = reader.GetSize()  # (X, Y, Z)
                spacing = list(reversed(reader.GetSpacing()))  # Z, Y, X
                
                # Estimate resampled depth
                D = size[2]
                zoom_z = spacing[0] / target_spacing
                D_resampled = int(round(D * zoom_z))
                
                num_chunks = max(1, int(np.ceil(D_resampled / crop_size)))
                for c in range(num_chunks):
                    self.chunks.append({
                        "series_idx": i,
                        "z_start": c * crop_size
                    })
            except Exception as e:
                print(f"  Warning: Skipping {info['uid']}: {e}")
        
        print(f"Dataset: {len(self.series_infos)} scans, {len(self.chunks)} chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def _load_scan(self, series_idx):
        """Load and resample scan with caching."""
        cached = self._cache.get(series_idx)
        if cached is not None:
            return cached
        
        info = self.series_infos[series_idx]
        uid = info['uid']
        
        # Load CT
        img_arr, origin, spacing = load_scan(info['path'])
        img_arr = normalize(img_arr)
        
        # Resample to isotropic
        img_arr, new_spacing = resample_isotropic(img_arr, spacing, self.target_spacing, order=1)
        
        # Load lung mask
        mask_path = os.path.join(self.seg_dir, uid + ".mhd")
        if os.path.exists(mask_path):
            mask_itk = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_itk)
            mask_arr, _ = resample_isotropic(mask_arr, spacing, self.target_spacing, order=0)
            mask_arr = (mask_arr > 0).astype(np.float32)
        else:
            mask_arr = np.zeros_like(img_arr, dtype=np.float32)
        
        result = (img_arr, mask_arr)
        self._cache.put(series_idx, result)
        return result
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        series_idx = chunk['series_idx']
        z_start = chunk['z_start']
        
        img_arr, mask_arr = self._load_scan(series_idx)
        
        # Extract chunk
        D = img_arr.shape[0]
        z_end = min(z_start + self.crop_size, D)
        img_chunk = img_arr[z_start:z_end]
        mask_chunk = mask_arr[z_start:z_end]
        
        # Pad if needed
        actual_d = img_chunk.shape[0]
        if actual_d < self.crop_size:
            pad = self.crop_size - actual_d
            img_chunk = np.pad(img_chunk, ((0, pad), (0, 0), (0, 0)), constant_values=0)
            mask_chunk = np.pad(mask_chunk, ((0, pad), (0, 0), (0, 0)), constant_values=0)
        
        # Resize H, W to 256x256 for consistent memory usage
        H, W = img_chunk.shape[1], img_chunk.shape[2]
        if H != 256 or W != 256:
            zoom_y = 256 / H
            zoom_x = 256 / W
            img_chunk = scipy.ndimage.zoom(img_chunk, (1.0, zoom_y, zoom_x), order=1)
            mask_chunk = scipy.ndimage.zoom(mask_chunk, (1.0, zoom_y, zoom_x), order=0)
            mask_chunk = (mask_chunk > 0.5).astype(np.float32)
        
        # To tensor: (1, D, H, W)
        image = torch.from_numpy(img_chunk).unsqueeze(0).float()
        mask = torch.from_numpy(mask_chunk).unsqueeze(0).float()
        
        return {"image": image, "mask": mask}


# ============================================================
#  Stage 2: Candidate Classification Dataset
# ============================================================

class CandidateDataset(Dataset):
    """Dataset for nodule candidate classification.
    
    Reads candidates_V2.csv, crops 32^3 isotropic patches
    around each candidate location.
    """
    
    def __init__(self, data_dir, candidates_file, subset_indices, 
                 patch_size=32, target_spacing=1.0, is_train=True):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.is_train = is_train
        self._cache = ScanCache(max_size=4)
        
        # Load candidates
        all_candidates = pd.read_csv(candidates_file)
        
        # Map seriesuid -> file path
        self.uid_to_path = {}
        for subset_idx in subset_indices:
            subset_dir = os.path.join(data_dir, f"subset{subset_idx}", f"subset{subset_idx}")
            if not os.path.isdir(subset_dir):
                continue
            for f in sorted(os.listdir(subset_dir)):
                if f.endswith(".mhd"):
                    uid = f.replace(".mhd", "")
                    self.uid_to_path[uid] = os.path.join(subset_dir, f)
        
        # Filter candidates to only those in our subsets
        available_uids = set(self.uid_to_path.keys())
        self.candidates = all_candidates[all_candidates['seriesuid'].isin(available_uids)].reset_index(drop=True)
        
        n_pos = (self.candidates['class'] == 1).sum()
        n_neg = (self.candidates['class'] == 0).sum()
        print(f"CandidateDataset: {len(self.candidates)} candidates "
              f"({n_pos} positive, {n_neg} negative) from {len(available_uids)} scans")
    
    def __len__(self):
        return len(self.candidates)
    
    def get_sample_weights(self):
        """Return per-sample weights for balanced sampling."""
        labels = self.candidates['class'].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        
        # Weight each class inversely proportional to frequency
        w_pos = len(labels) / (2.0 * n_pos)
        w_neg = len(labels) / (2.0 * n_neg)
        
        weights = np.where(labels == 1, w_pos, w_neg)
        return torch.from_numpy(weights).float()
    
    def _load_scan(self, uid):
        """Load and resample scan with caching."""
        cached = self._cache.get(uid)
        if cached is not None:
            return cached
        
        path = self.uid_to_path[uid]
        img_arr, origin, spacing = load_scan(path)
        img_arr = normalize(img_arr)
        
        # Resample to isotropic
        img_arr, new_spacing = resample_isotropic(img_arr, spacing, self.target_spacing, order=1)
        new_origin = origin  # Origin stays in world coords
        
        result = (img_arr, new_origin, spacing)  # Keep original spacing for world->voxel conversion
        self._cache.put(uid, result)
        return result
    
    def __getitem__(self, idx):
        row = self.candidates.iloc[idx]
        uid = row['seriesuid']
        label = int(row['class'])
        
        # World coordinates (X, Y, Z in CSV -> Z, Y, X for numpy)
        center_world = np.array([row['coordZ'], row['coordY'], row['coordX']])
        
        img_arr, origin, orig_spacing = self._load_scan(uid)
        
        # Convert world -> voxel in isotropic space
        target_sp = np.array([self.target_spacing] * 3)
        center_vox = world_to_voxel(center_world, origin, target_sp)
        
        # Crop patch centered on candidate
        half = self.patch_size // 2
        D, H, W = img_arr.shape
        
        cz, cy, cx = int(round(center_vox[0])), int(round(center_vox[1])), int(round(center_vox[2]))
        
        # Compute slice boundaries with padding
        z1, z2 = cz - half, cz + half
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        
        # Clamp to valid range
        sz1, sz2 = max(0, z1), min(D, z2)
        sy1, sy2 = max(0, y1), min(H, y2)
        sx1, sx2 = max(0, x1), min(W, x2)
        
        # Extract and pad
        patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
        
        # Offsets into the patch array
        pz1 = sz1 - z1
        py1 = sy1 - y1
        px1 = sx1 - x1
        pz2 = pz1 + (sz2 - sz1)
        py2 = py1 + (sy2 - sy1)
        px2 = px1 + (sx2 - sx1)
        
        patch[pz1:pz2, py1:py2, px1:px2] = img_arr[sz1:sz2, sy1:sy2, sx1:sx2]
        
        # Data augmentation (training only)
        if self.is_train:
            # Random flips
            if np.random.rand() > 0.5:
                patch = patch[::-1].copy()  # Flip Z
            if np.random.rand() > 0.5:
                patch = patch[:, ::-1].copy()  # Flip Y
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()  # Flip X
        
        # To tensor: (1, D, H, W)
        patch_tensor = torch.from_numpy(patch.copy()).unsqueeze(0).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return {"patch": patch_tensor, "label": label_tensor}


# ============================================================
#  Fold Loaders
# ============================================================

def get_seg_fold_loaders(data_dir, fold, total_folds=5, crop_size=64,
                         num_workers=4, batch_size=1, target_spacing=1.0):
    """Get train/val loaders for lung segmentation (Stage 1)."""
    
    all_subsets = list(range(total_folds))
    val_subsets = [all_subsets[fold]]
    train_subsets = [s for s in all_subsets if s != fold]
    
    seg_dir = os.path.join(data_dir, "seg-lungs-LUNA16", "seg-lungs-LUNA16")
    
    print(f"Fold {fold+1}/{total_folds}:")
    print(f"  Train: subsets {train_subsets}")
    print(f"  Val:   subsets {val_subsets}")
    
    train_dataset = LungSegDataset(
        data_dir=data_dir, subset_indices=train_subsets,
        seg_dir=seg_dir, crop_size=crop_size,
        target_spacing=target_spacing, is_train=True
    )
    
    val_dataset = LungSegDataset(
        data_dir=data_dir, subset_indices=val_subsets,
        seg_dir=seg_dir, crop_size=crop_size,
        target_spacing=target_spacing, is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, train_subsets, val_subsets


def _build_scan_sorted_loader(dataset, indices, batch_size, num_workers, shuffle_scans=False):
    """Build a DataLoader where items are sorted by scan for cache locality."""
    from torch.utils.data import Subset
    
    # Sort indices by seriesuid for cache hits
    uids = dataset.candidates.iloc[indices]['seriesuid'].values
    
    if shuffle_scans:
        # Shuffle at scan level (not sample level) for some randomness
        unique_uids = np.unique(uids)
        np.random.shuffle(unique_uids)
        sorted_indices = []
        for uid in unique_uids:
            mask = uids == uid
            sorted_indices.extend(indices[mask].tolist())
        indices = np.array(sorted_indices)
    else:
        indices = indices[np.argsort(uids)]
    
    subset = Subset(dataset, indices.tolist())
    
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0
    )
    return loader


def get_candidate_fold_loaders(data_dir, candidates_file, fold, total_folds=5,
                                patch_size=32, num_workers=4, batch_size=16,
                                target_spacing=1.0, neg_ratio=1):
    """Get train/val loaders for candidate classification (Stage 2).
    
    Returns:
        train_dataset, val_loader, train_subsets, val_subsets, resample_fn
        
        Call resample_fn(epoch) at the start of each epoch to get a fresh
        train_loader with newly sampled negatives.
    """
    
    all_subsets = list(range(total_folds))
    val_subsets = [all_subsets[fold]]
    train_subsets = [s for s in all_subsets if s != fold]
    
    print(f"Fold {fold+1}/{total_folds}:")
    print(f"  Train: subsets {train_subsets}")
    print(f"  Val:   subsets {val_subsets}")
    
    train_dataset = CandidateDataset(
        data_dir=data_dir, candidates_file=candidates_file,
        subset_indices=train_subsets, patch_size=patch_size,
        target_spacing=target_spacing, is_train=True
    )
    
    val_dataset = CandidateDataset(
        data_dir=data_dir, candidates_file=candidates_file,
        subset_indices=val_subsets, patch_size=patch_size,
        target_spacing=target_spacing, is_train=False
    )
    
    # Pre-compute positive/negative indices for training
    labels = train_dataset.candidates['class'].values
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    n_pos = len(pos_indices)
    n_neg_sample = min(n_pos * neg_ratio, len(neg_indices))
    
    print(f"  Training: {n_pos} pos, {len(neg_indices)} neg total")
    print(f"  Per epoch: {n_pos} pos + {n_neg_sample} neg = {n_pos + n_neg_sample} samples")
    print(f"  ~{len(neg_indices) // max(n_neg_sample, 1)} epochs to see all negatives")
    
    def resample_train_loader(epoch):
        """Call at the start of each epoch to get a new loader with fresh negatives."""
        rng = np.random.default_rng(seed=fold * 1000 + epoch)
        neg_sampled = rng.choice(neg_indices, size=n_neg_sample, replace=False)
        balanced = np.concatenate([pos_indices, neg_sampled])
        return _build_scan_sorted_loader(
            train_dataset, balanced, batch_size, num_workers, shuffle_scans=True
        )
    
    # ---- Val: fixed subset, 10:1 neg:pos for realistic eval ----
    val_labels = val_dataset.candidates['class'].values
    val_pos = np.where(val_labels == 1)[0]
    val_neg = np.where(val_labels == 0)[0]
    n_val_neg = min(len(val_pos) * 10, len(val_neg))
    rng = np.random.default_rng(seed=fold)
    val_neg_sampled = rng.choice(val_neg, size=n_val_neg, replace=False)
    val_balanced = np.concatenate([val_pos, val_neg_sampled])
    
    val_loader = _build_scan_sorted_loader(
        val_dataset, val_balanced, batch_size, num_workers, shuffle_scans=False
    )
    print(f"  Validation: {len(val_pos)} pos + {n_val_neg} neg = {len(val_balanced)} samples")
    
    return train_dataset, val_loader, train_subsets, val_subsets, resample_train_loader
