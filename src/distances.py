# -*- coding: utf-8 -*-
"""
BOB DISTANCE MATRICES
Distance functions (Hungarian-L2, Chamfer, Hungarian-Cosine, OT),
distance matrix builders, and cluster label utilities.
"""

import os
import re
from typing import Optional

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .config import (
    SEED, OUTPUT_DIR, HAS_POT,
)


def hungarian_l2(vocab1: np.ndarray, vocab2: np.ndarray) -> float:
    C    = cdist(vocab1, vocab2, 'euclidean')
    r, c = linear_sum_assignment(C)
    return C[r, c].sum() / vocab1.shape[0]


def chamfer(vocab1: np.ndarray, vocab2: np.ndarray) -> float:
    C = cdist(vocab1, vocab2, 'euclidean')
    return (np.mean(np.min(C, axis=1)) + np.mean(np.min(C, axis=0))) / 2.0


def hungarian_cosine(vocab1: np.ndarray, vocab2: np.ndarray) -> float:
    v1   = vocab1 / (np.linalg.norm(vocab1, axis=1, keepdims=True) + 1e-8)
    v2   = vocab2 / (np.linalg.norm(vocab2, axis=1, keepdims=True) + 1e-8)
    C    = cdist(v1, v2, 'cosine')
    r, c = linear_sum_assignment(C)
    return C[r, c].sum() / vocab1.shape[0]


BOB_DISTANCE_FNS = {
    'hungarian_l2':     hungarian_l2,
    'chamfer':          chamfer,
    'hungarian_cosine': hungarian_cosine,
}


def build_bob_distance_matrix(page_data: dict, metric: str = 'hungarian_l2') -> tuple:
    """
    Build the N×N BoB pairwise distance matrix for the given metric.
    Returns (distance_matrix, page_names).
    """
    page_names = sorted(page_data.keys())
    dist_fn    = BOB_DISTANCE_FNS[metric]

    n  = len(page_names)
    dm = np.zeros((n, n))
    print(f"\nBuilding {n}x{n} BoB distance matrix ({metric})...")
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            d        = dist_fn(page_data[page_names[i]]['vocabulary'],
                               page_data[page_names[j]]['vocabulary'])
            dm[i,j]  = dm[j,i] = d

    off = dm[np.triu_indices(n, k=1)]
    print(f"  min={off.min():.4f}  mean={off.mean():.4f}  "
          f"median={np.median(off):.4f}  max={off.max():.4f}")
    return dm, page_names


def bob_ot_weighted(vocab1: np.ndarray, cluster_data1: list,
                    vocab2: np.ndarray, cluster_data2: list) -> float:
    """
    Weighted Wasserstein-1: visual word i has weight proportional to
    the number of patches assigned to it (cluster population).
    Uses POT if available; falls back to uniform Hungarian otherwise.
    """
    w1 = np.array([max(1, len(c['patch_locations'])) for c in cluster_data1], dtype=float)
    w2 = np.array([max(1, len(c['patch_locations'])) for c in cluster_data2], dtype=float)
    w1 /= w1.sum()
    w2 /= w2.sum()
    M  = cdist(vocab1, vocab2, 'euclidean')

    if HAS_POT:
        import ot
        T = ot.emd(w1, w2, M)
        return float(np.sum(T * M))
    # Fallback: uniform Hungarian
    r, c = linear_sum_assignment(M)
    return float(M[r, c].sum() / len(w1))


def _build_ot_matrix(page_data: dict, page_names: list) -> tuple:
    """Build the BoB-OT (weighted) N×N distance matrix."""
    n  = len(page_names)
    dm = np.zeros((n, n))
    print(f"\nBuilding {n}×{n} BoB-OT (weighted) matrix...")
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            d = bob_ot_weighted(
                page_data[page_names[i]]['vocabulary'],
                page_data[page_names[i]]['clusters_data'],
                page_data[page_names[j]]['vocabulary'],
                page_data[page_names[j]]['clusters_data'],
            )
            dm[i, j] = dm[j, i] = d
    print(f"BoB-OT matrix built.")
    return dm, page_names


def build_bob_ot_matrix(page_data: dict) -> tuple:
    """Build the BoB-OT weighted distance matrix."""
    page_names = sorted(page_data.keys())
    return _build_ot_matrix(page_data, page_names)


def get_cluster_label(img_name: str) -> str:
    """
    Works for both basename keys and relative-path page_id keys.
    """
    base_name = os.path.basename(img_name)
    base = os.path.splitext(base_name)[0]

    m = re.match(r'(cluster_\d+)', base)
    if m:
        return m.group(1)

    m2 = re.match(r'(.*?)__L\d', base)
    if m2:
        return m2.group(1)

    return base
