# -*- coding: utf-8 -*-
"""
BCC GRAPH ANALYSIS
Bi-connected component detection, similarity graph construction,
color image copying, and matched-pair figure generation per cluster.
"""

import os
import itertools
import shutil
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .config import (
    VOCAB_SIZE, MIN_AREA, MAX_AREA, COLOR_IMAGE_DIR, FEAT_IMAGE_DIR,
    BCC_OUTPUT_DIR, SIMILARITY_THRESHOLD, MIN_BCC_SIZE,
    N_PAIRS_PER_CLUSTER, NUM_SAMPLES_TO_PLOT,
)
from .distances import get_cluster_label
from .utils import _image_cache, load_patch_for_viz


def _save_matched_pairs_figure(page1: str, page2: str, page_data: dict,
                                n_samples: int, out_path: str):
    """Render a matched visual word figure and save to disk (no plt.show)."""
    vocab1 = page_data[page1]['vocabulary']
    vocab2 = page_data[page2]['vocabulary']
    C      = cdist(vocab1, vocab2, 'euclidean')
    r, c   = linear_sum_assignment(C)
    costs  = C[r, c]
    order  = np.argsort(costs)

    ncols = n_samples * 2 + 1
    fig, axes = plt.subplots(VOCAB_SIZE, ncols,
                              figsize=(ncols * 1.5, VOCAB_SIZE * 2.5))
    fig.suptitle(f"Matched Vocabulary\n{page1}\n{page2}", fontsize=12, y=1.02)

    for row, pidx in enumerate(order):
        vw1, vw2 = r[pidx], c[pidx]
        locs1 = page_data[page1]['clusters_data'][vw1]['patch_locations']
        locs2 = page_data[page2]['clusters_data'][vw2]['patch_locations']

        axes[row, 0].set_ylabel(
            f"R{row+1} {costs[pidx]:.2f}\nVW{vw1}|VW{vw2}",
            rotation=0, labelpad=45, va='center', fontsize=6)

        for j, loc in enumerate(random.sample(locs1, min(len(locs1), n_samples))):
            axes[row, j].imshow(
                load_patch_for_viz(loc['img_path'], loc['coords']), cmap='gray')
            if row == 0: axes[row, j].set_title(f"P1-{j+1}", fontsize=7)

        axes[row, n_samples].set_visible(False)

        for j, loc in enumerate(random.sample(locs2, min(len(locs2), n_samples))):
            axes[row, j + n_samples + 1].imshow(
                load_patch_for_viz(loc['img_path'], loc['coords']), cmap='gray')
            if row == 0: axes[row, j + n_samples + 1].set_title(f"P2-{j+1}", fontsize=7)

        for j in range(ncols):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def _load_page_image_for_grid(page_name: str, page_data: dict) -> np.ndarray:
    """Load page image for retrieval grids. Uses stored binary image path."""
    img_path = page_data[page_name].get('img_path', page_name)
    full_path = img_path if os.path.exists(img_path) else os.path.join(FEAT_IMAGE_DIR, os.path.basename(img_path))

    img = np.array(Image.open(full_path).convert('L'))
    if np.mean(img) > 128:
        img = 255 - img
    return img


def plot_query_topk_grid(query_name: str, distance_matrix: np.ndarray, page_names: list,
                         page_data: dict, label_fn=get_cluster_label, k: int = 5,
                         save_name: str = None):
    """
    Show one query and its top-k retrievals with correct/incorrect titles.
    Stronger than PCA for the paper.
    """
    from .config import OUTPUT_DIR
    if query_name not in page_names:
        raise ValueError(f"{query_name} not found in page_names")

    qi = page_names.index(query_name)
    d = distance_matrix[qi].copy()
    d[qi] = np.inf
    ranked = np.argsort(d)[:k]
    gt = label_fn(query_name)

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))

    axes[0].imshow(_load_page_image_for_grid(query_name, page_data), cmap='gray')
    axes[0].set_title(f'Query\n{query_name[:28]}', fontsize=9)
    axes[0].axis('off')

    for col, j in enumerate(ranked, 1):
        name = page_names[j]
        pred_ok = label_fn(name) == gt

        axes[col].imshow(_load_page_image_for_grid(name, page_data), cmap='gray')
        axes[col].set_title(
            f"Rank {col} | {'✓' if pred_ok else '✗'}\n{d[j]:.3f}",
            fontsize=9
        )

        for spine in axes[col].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor('green' if pred_ok else 'red')

        axes[col].axis('off')

    plt.tight_layout()
    save_name = save_name or f"query_top{k}_{os.path.splitext(query_name)[0]}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def run_bcc_analysis(df_matrix: pd.DataFrame, page_data: dict,
                     threshold: float = SIMILARITY_THRESHOLD):
    """
    Build a similarity graph from the distance matrix, find bi-connected components
    of size >= MIN_BCC_SIZE, copy color images and save matched-pair plots per cluster.
    """
    all_names = list(df_matrix.index)
    G = nx.Graph()
    G.add_nodes_from(all_names)

    print(f"\nBuilding similarity graph (threshold={threshold})...")
    for i in tqdm(range(len(all_names))):
        for j in range(i+1, len(all_names)):
            if df_matrix.iloc[i, j] < threshold:
                G.add_edge(all_names[i], all_names[j])
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    bccs = sorted(
        [b for b in nx.biconnected_components(G) if len(b) >= MIN_BCC_SIZE],
        key=len, reverse=True)
    print(f"Found {len(bccs)} BCCs with size >= {MIN_BCC_SIZE}")
    os.makedirs(BCC_OUTPUT_DIR, exist_ok=True)

    for idx, bcc_set in enumerate(bccs, 1):
        bcc_pages   = sorted(bcc_set)
        cluster_dir = os.path.join(BCC_OUTPUT_DIR, f"bcc_{idx:03d}_size{len(bcc_pages)}")
        compare_dir = os.path.join(cluster_dir, '_comparisons')
        os.makedirs(compare_dir, exist_ok=True)
        _image_cache.clear()

        print(f"\nBCC {idx} (size {len(bcc_pages)})")

        # Copy color images
        for page in bcc_pages:
            src = os.path.join(COLOR_IMAGE_DIR,
                               os.path.splitext(page)[0].replace('_bin', '_col') + '.jpg')
            dst = os.path.join(cluster_dir, os.path.basename(src))
            if not os.path.exists(dst) and os.path.exists(src):
                shutil.copy(src, dst)

        # Collect direct-edge pairs, prioritize cross-library
        cross, intra = [], []
        for p1, p2 in itertools.combinations(bcc_pages, 2):
            d = df_matrix.loc[p1, p2]
            if d >= threshold:
                continue
            (cross if p1.split('__')[0] != p2.split('__')[0] else intra).append((d, p1, p2))
        cross.sort(); intra.sort()
        pairs = (cross + intra)[:N_PAIRS_PER_CLUSTER]

        for d, p1, p2 in tqdm(pairs, desc=f"  Plots BCC {idx}"):
            s1, s2   = os.path.splitext(p1)[0], os.path.splitext(p2)[0]
            out_path = os.path.join(compare_dir, f"compare_{s1}_VS_{s2}_d{d:.3f}.png")
            if os.path.exists(out_path):
                continue
            try:
                _save_matched_pairs_figure(p1, p2, page_data, NUM_SAMPLES_TO_PLOT, out_path)
            except Exception as e:
                print(f"  Plot error {p1} vs {p2}: {e}")

    print("\nBCC analysis complete.")
