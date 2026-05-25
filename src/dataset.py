# -*- coding: utf-8 -*-

import os
import glob
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import (
    SEED, DEVICE, PATCH_SIZE, ENCODED_DIM, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, SPARSITY_WEIGHT, PATCHES_PER_IMAGE_TRAIN,
    MIN_AREA, MAX_AREA, MIN_VALID_COMPONENTS, TARGET_FIT_SIZE,
    MIN_COMPONENT_WHITE_RATIO, MIN_CANVAS_WHITE_RATIO,
    TRAIN_PATCH_MODE, OUTPUT_DIR,
)
from .model import SparseAutoencoder


@dataclass
class PatchFilterConfig:
    min_area: int = MIN_AREA
    max_area: int = MAX_AREA
    min_component_white_ratio: float = MIN_COMPONENT_WHITE_RATIO
    min_canvas_white_ratio: float = MIN_CANVAS_WHITE_RATIO
    min_valid_components: int = MIN_VALID_COMPONENTS
    target_fit_size: int = TARGET_FIT_SIZE


def _list_image_paths(folder_path: str) -> List[str]:
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    return sorted(image_paths)


def _load_binary_image(img_path: str) -> np.ndarray:
    img_array = np.array(Image.open(img_path).convert('L'))
    # Ensure text = white
    if np.mean(img_array) > 128:
        img_array = 255 - img_array
    return img_array


def _make_page_id(img_path: str, dataset_root: Optional[str] = None) -> str:
    if dataset_root is None:
        return os.path.normpath(img_path)
    return os.path.normpath(os.path.relpath(img_path, dataset_root))


def _context_patch_from_centroid(img_array: np.ndarray, cy: int, cx: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    padded = np.pad(img_array, half, constant_values=0)
    patch = padded[cy: cy + patch_size, cx: cx + patch_size]
    return patch


def _fit_and_pad_patch(component_crop: np.ndarray, patch_size: int) -> np.ndarray:
    """Scale component so its longest side equals TARGET_FIT_SIZE, then center-pad."""
    h, w    = component_crop.shape[:2]
    max_dim = max(w, h)
    if max_dim == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    scale   = TARGET_FIT_SIZE / max_dim
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(component_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((patch_size, patch_size), dtype=np.uint8)
    py = (patch_size - new_h) // 2
    px = (patch_size - new_w) // 2
    canvas[py: py + new_h, px: px + new_w] = resized
    return canvas


def extract_component_records(
    img_path: str,
    mode: str,
    patch_size: int,
    filter_cfg: PatchFilterConfig,
    dataset_root: Optional[str] = None,
    patches_per_img: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[dict], dict]:
    """
    Shared extractor used by BOTH:
      - TextPatchDataset (training)
      - get_image_features (inference / retrieval)

    mode='context' : centroid-centered fixed windows
    mode='fitpad'  : component bbox -> fit-and-pad normalized canvas

    Returns:
      records: list of dicts with 'patch' and metadata
      summary: per-page accounting for filtering / exclusions
    """
    if mode not in ('context', 'fitpad'):
        raise ValueError(f"Unknown mode: {mode}")

    summary = {
        'img_path': img_path,
        'page_id': _make_page_id(img_path, dataset_root),
        'mode': mode,
        'num_components_total': 0,
        'num_area_valid': 0,
        'num_component_white_valid': 0,
        'num_canvas_white_valid': 0,
        'num_final_valid': 0,
        'excluded': False,
        'exclusion_reason': '',
    }

    try:
        img_array = _load_binary_image(img_path)
    except Exception as e:
        summary['excluded'] = True
        summary['exclusion_reason'] = f'load_error: {e}'
        return [], summary

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(img_array, connectivity=8)
    if num_labels <= 1:
        summary['excluded'] = True
        summary['exclusion_reason'] = 'no_components'
        return [], summary

    summary['num_components_total'] = int(num_labels - 1)

    candidate_records = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (filter_cfg.min_area <= area <= filter_cfg.max_area):
            continue
        summary['num_area_valid'] += 1

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        component_crop = img_array[y:y+h, x:x+w]
        component_white_ratio = float(np.mean(component_crop > 128))
        if component_white_ratio < filter_cfg.min_component_white_ratio:
            continue
        summary['num_component_white_valid'] += 1

        cy = int(centroids[i, 1])
        cx = int(centroids[i, 0])

        if mode == 'context':
            patch = _context_patch_from_centroid(img_array, cy, cx, patch_size)
        else:
            patch = _fit_and_pad_patch(component_crop, patch_size)

        if patch.shape != (patch_size, patch_size):
            continue

        canvas_white_ratio = float(np.mean(patch > 128))
        if canvas_white_ratio < filter_cfg.min_canvas_white_ratio:
            continue
        summary['num_canvas_white_valid'] += 1

        candidate_records.append({
            'patch': patch,
            'meta': {
                'page_id': _make_page_id(img_path, dataset_root),
                'img_path': img_path,
                'coords': (cy, cx),
                'bbox': (x, y, w, h),
                'component_idx': i,
                'area': area,
                'component_white_ratio': component_white_ratio,
                'canvas_white_ratio': canvas_white_ratio,
                'patch_mode': mode,
            }
        })

    if mode == 'context' and patches_per_img is not None and len(candidate_records) > patches_per_img:
        rng = rng or random.Random(SEED)
        keep_idx = sorted(rng.sample(range(len(candidate_records)), patches_per_img))
        candidate_records = [candidate_records[j] for j in keep_idx]

    summary['num_final_valid'] = len(candidate_records)

    if len(candidate_records) < filter_cfg.min_valid_components:
        summary['excluded'] = True
        summary['exclusion_reason'] = f'too_few_valid_components(<{filter_cfg.min_valid_components})'

    return candidate_records, summary


class TextPatchDataset(Dataset):
    """
    Training dataset built from the SAME shared component extractor used at inference.
    Supports:
      - patch_mode='context'
      - patch_mode='fitpad'
    """
    def __init__(
        self,
        folder_path: str,
        patch_size: int,
        patches_per_img: int,
        transform=None,
        patch_mode: str = TRAIN_PATCH_MODE,
        filter_cfg: Optional[PatchFilterConfig] = None,
        dataset_root: Optional[str] = None,
        save_manifest_path: Optional[str] = None,
    ):
        self.patch_size = patch_size
        self.transform = transform
        self.patch_mode = patch_mode
        self.filter_cfg = filter_cfg or PatchFilterConfig()
        self.dataset_root = dataset_root or folder_path

        self.all_patches = []
        self.patch_meta = []
        self.page_manifest = []

        image_paths = _list_image_paths(folder_path)
        print(f"Found {len(image_paths)} images for training. mode={patch_mode}")

        rng = random.Random(SEED)

        for img_path in tqdm(image_paths, desc=f"Building patch dataset [{patch_mode}]"):
            records, summary = extract_component_records(
                img_path=img_path,
                mode=patch_mode,
                patch_size=patch_size,
                filter_cfg=self.filter_cfg,
                dataset_root=self.dataset_root,
                patches_per_img=patches_per_img if patch_mode == 'context' else patches_per_img,
                rng=rng,
            )
            self.page_manifest.append(summary)

            if summary['excluded']:
                continue

            for rec in records:
                self.all_patches.append(rec['patch'])
                self.patch_meta.append(rec['meta'])

        print(f"Collected {len(self.all_patches)} training patches.")
        print(f"Included pages: {sum(not r['excluded'] for r in self.page_manifest)} / {len(self.page_manifest)}")

        if save_manifest_path is not None:
            pd.DataFrame(self.page_manifest).to_csv(save_manifest_path, index=False)

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        patch = Image.fromarray(self.all_patches[idx])
        return self.transform(patch) if self.transform else transforms.ToTensor()(patch)


def train_autoencoder(
    image_dir: str,
    patch_mode: str = TRAIN_PATCH_MODE,
    model_save_path: Optional[str] = None,
    recon_save_path: Optional[str] = None,
    manifest_save_path: Optional[str] = None,
) -> SparseAutoencoder:
    filter_cfg = PatchFilterConfig()

    dataset = TextPatchDataset(
        folder_path=image_dir,
        patch_size=PATCH_SIZE,
        patches_per_img=PATCHES_PER_IMAGE_TRAIN,
        transform=transforms.ToTensor(),
        patch_mode=patch_mode,
        filter_cfg=filter_cfg,
        dataset_root=image_dir,
        save_manifest_path=manifest_save_path,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No training patches found for patch_mode={patch_mode}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    viz_batch = next(iter(loader))[:8].to(DEVICE)

    model = SparseAutoencoder(ENCODED_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining autoencoder for {EPOCHS} epochs... patch_mode={patch_mode}")
    for epoch in range(EPOCHS):
        model.train()
        total, recon_sum, l1_sum = 0.0, 0.0, 0.0

        for patches in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            patches = patches.to(DEVICE)
            encoded, decoded = model(patches)
            recon = criterion(decoded, patches)
            l1 = torch.sum(torch.abs(encoded)) / encoded.size(0)
            loss = recon + SPARSITY_WEIGHT * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            recon_sum += recon.item()
            l1_sum += l1.item()

        n = len(loader)
        print(f"  Epoch {epoch+1:>2}/{EPOCHS}  loss={total/n:.5f}  recon={recon_sum/n:.5f}  L1={l1_sum/n:.5f}")

    if model_save_path is None:
        model_save_path = os.path.join(OUTPUT_DIR, f'bob_model_{patch_mode}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    model.eval()
    with torch.no_grad():
        _, recons = model(viz_batch)

    grid = vutils.make_grid(
        torch.cat([viz_batch.cpu(), recons.cpu()]),
        nrow=8, padding=2, normalize=True
    )
    plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(f"Originals (top) vs Reconstructions (bottom) [{patch_mode}]")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))

    if recon_save_path is None:
        recon_save_path = os.path.join(OUTPUT_DIR, f'fig_reconstructions_{patch_mode}.png')
    plt.savefig(recon_save_path, dpi=150)
    plt.show()

    return model
