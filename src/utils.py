# -*- coding: utf-8 -*-
"""
PATCH UTILITIES
Shared image loading/caching and patch reconstruction utilities
used by visualization and BCC analysis.
"""

import os

import numpy as np
import cv2
from PIL import Image

from .config import (
    PATCH_SIZE, MIN_AREA, MAX_AREA, FEAT_IMAGE_DIR,
)
from .dataset import _fit_and_pad_patch


# Module-level image cache shared across visualization and BCC modules
_image_cache: dict = {}


def _get_image_stats(img_path: str, image_dir: str = FEAT_IMAGE_DIR):
    """
    Load a binary image and run connectedComponentsWithStats (cached by full path).
    Uses img_path directly if it exists (full absolute path stored in metadata),
    otherwise falls back to joining with image_dir.
    Auto-inverts if image is black-text-on-white.
    """
    # Prefer the stored absolute path; fall back to flat join for legacy data
    full_path = img_path if os.path.exists(img_path) \
                else os.path.join(image_dir, os.path.basename(img_path))

    if full_path in _image_cache:
        return _image_cache[full_path]

    try:
        img_array = np.array(Image.open(full_path).convert('L'))

        # Match the same polarity correction used in get_image_features
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            img_array, connectivity=8)
        num_labels = stats.shape[0]
        _image_cache[full_path] = (img_array, stats, centroids, num_labels)
        return _image_cache[full_path]
    except Exception as e:
        print(f"  _get_image_stats failed for {full_path}: {e}")
        return None, None, None, 0


def load_patch_for_viz(img_path: str, center_coords: tuple,
                       image_dir: str = FEAT_IMAGE_DIR) -> Image.Image:
    """
    Reconstruct the fit-and-padded patch nearest to `center_coords` in the given image.
    Returns a blank patch on failure.
    """
    blank = Image.fromarray(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))
    try:
        ty, tx = center_coords
        img_array, stats, centroids, num_labels = _get_image_stats(img_path, image_dir)
        if img_array is None:
            return blank

        valid = [i for i in range(1, num_labels)
                 if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA]
        if not valid:
            return blank

        dists  = np.linalg.norm(centroids[valid] - [tx, ty], axis=1)
        best_i = valid[np.argmin(dists)]
        x = stats[best_i, cv2.CC_STAT_LEFT];  y = stats[best_i, cv2.CC_STAT_TOP]
        w = stats[best_i, cv2.CC_STAT_WIDTH]; h = stats[best_i, cv2.CC_STAT_HEIGHT]
        return Image.fromarray(_fit_and_pad_patch(img_array[y: y+h, x: x+w], PATCH_SIZE))
    except Exception:
        return blank
