# -*- coding: utf-8 -*-
"""
VISUALIZATIONS
Matched pair visualization and distance distribution plots.
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

from .config import (
    SEED, OUTPUT_DIR, PATCH_SIZE, VOCAB_SIZE,
)
from .distances import get_cluster_label
from .utils import load_patch_for_viz, _image_cache


def visualize_matched_pairs(page_name_1: str, page_name_2: str,
                             page_data: dict, df_matrix: pd.DataFrame,
                             n_samples: int = 5):
    """
    Show the VOCAB_SIZE Hungarian-matched visual word pairs between two pages,
    ranked by match cost (best match first).
    """
    if page_name_1 not in page_data or page_name_2 not in page_data:
        print("Error: page(s) not found in page_data.")
        return

    _image_cache.clear()
    print(f"Overall BoB distance: {df_matrix.loc[page_name_1, page_name_2]:.4f}")

    vocab1 = page_data[page_name_1]['vocabulary']
    vocab2 = page_data[page_name_2]['vocabulary']
    C      = cdist(vocab1, vocab2, 'euclidean')
    r, c   = linear_sum_assignment(C)
    costs  = C[r, c]
    order  = np.argsort(costs)

    ncols = n_samples * 2 + 1
    fig, axes = plt.subplots(VOCAB_SIZE, ncols,
                              figsize=(ncols * 1.5, VOCAB_SIZE * 2.5))
    fig.suptitle(f"Matched Vocabulary\n{page_name_1}\n{page_name_2}",
                 fontsize=14, y=1.02)

    for row, pair_idx in enumerate(order):
        vw1, vw2, cost = r[pair_idx], c[pair_idx], costs[pair_idx]
        locs1 = page_data[page_name_1]['clusters_data'][vw1]['patch_locations']
        locs2 = page_data[page_name_2]['clusters_data'][vw2]['patch_locations']

        axes[row, 0].set_ylabel(
            f"Rank {row+1}\nCost:{cost:.3f}\nVW{vw1}|VW{vw2}",
            rotation=0, labelpad=55, va='center', fontsize=7)

        for j, loc in enumerate(random.sample(locs1, min(len(locs1), n_samples))):
            ax = axes[row, j]
            ax.imshow(load_patch_for_viz(loc['img_path'], loc['coords']), cmap='gray')
            if row == 0: ax.set_title(f"P1-{j+1}", fontsize=8)

        axes[row, n_samples].set_visible(False)

        for j, loc in enumerate(random.sample(locs2, min(len(locs2), n_samples))):
            ax = axes[row, j + n_samples + 1]
            ax.imshow(load_patch_for_viz(loc['img_path'], loc['coords']), cmap='gray')
            if row == 0: ax.set_title(f"P2-{j+1}", fontsize=8)

        for j in range(ncols):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def compute_distance_separation_stats(dm: np.ndarray, names: list,
                                      label_fn=get_cluster_label,
                                      method_name: str = 'method',
                                      n_boot: int = 1000,
                                      seed: int = 42) -> dict:
    """
    Summarize within-method intra vs inter separation.
    Reports KS statistic, a rank-AUC-style separation score, Cohen's d,
    and bootstrap 95% CI for the mean gap (inter - intra).
    """
    labels = [label_fn(n) for n in names]
    intra, inter = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if labels[i] == labels[j]:
                intra.append(dm[i, j])
            else:
                inter.append(dm[i, j])

    intra = np.asarray(intra, dtype=float)
    inter = np.asarray(inter, dtype=float)

    if len(intra) == 0 or len(inter) == 0:
        return {'method': method_name}

    ks = ks_2samp(intra, inter)

    pooled = np.sqrt((np.var(intra, ddof=1) + np.var(inter, ddof=1)) / 2.0 + 1e-12)
    cohen_d = (inter.mean() - intra.mean()) / pooled

    # Rank-AUC-style separation: P(inter > intra) + 0.5 P(equal)
    comp = inter[:, None] - intra[None, :]
    auc_sep = np.mean(comp > 0) + 0.5 * np.mean(comp == 0)

    rng = np.random.default_rng(seed)
    boot_gap = []
    for _ in range(n_boot):
        intra_s = rng.choice(intra, size=len(intra), replace=True)
        inter_s = rng.choice(inter, size=len(inter), replace=True)
        boot_gap.append(inter_s.mean() - intra_s.mean())
    lo, hi = np.percentile(boot_gap, [2.5, 97.5])

    return {
        'method': method_name,
        'n_intra': int(len(intra)),
        'n_inter': int(len(inter)),
        'intra_mean': float(intra.mean()),
        'inter_mean': float(inter.mean()),
        'mean_gap': float(inter.mean() - intra.mean()),
        'gap_ci95_low': float(lo),
        'gap_ci95_high': float(hi),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
        'auc_sep': float(auc_sep),
        'cohen_d': float(cohen_d),
    }

