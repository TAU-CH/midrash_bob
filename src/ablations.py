# -*- coding: utf-8 -*-
"""
ABLATION STUDIES AND RUNTIME PROFILING: Vocabulary size ablation, encoded dimension ablation, sparsity ablation,
fit-and-pad ablation, patch mode ablation, and runtime profiling table.
"""

import os
import glob
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .config import (
    SEED, DEVICE, PATCH_SIZE, ENCODED_DIM, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, SPARSITY_WEIGHT, PATCHES_PER_IMAGE_TRAIN,
    MIN_AREA, MAX_AREA, MIN_VALID_COMPONENTS, MIN_CANVAS_WHITE_RATIO,
    VOCAB_SIZE, OUTPUT_DIR, INFER_PATCH_MODE, RERANK_TOP_M,
    ABLATION_K_VALUES, ABLATION_DIM_VALUES, ABLATION_DISPLAY_COLS,
    PATCH_MODE_ABLATIONS,
)
from .model import SparseAutoencoder
from .dataset import TextPatchDataset, _list_image_paths, _load_binary_image
from .features import _encode_patches, get_image_features, extract_all_features, build_bob_vocabularies
from .distances import hungarian_l2, get_cluster_label, bob_ot_weighted
from .evaluation import compute_retrieval_metrics
from .visualization import compute_distance_separation_stats
from .dataset import PatchFilterConfig


# ==============================================================================
# SECTION 12: ABLATION STUDIES
# ==============================================================================

def _cluster_cache(features_cache: dict, k: int) -> dict:
    """KMeans with vocab size k on pre-encoded features. Used by all ablations."""
    page_data = {}
    for img_name, (feats, metas) in tqdm(features_cache.items(),
                                          desc=f"KMeans k={k}", leave=False):
        if feats.shape[0] < k:
            continue
        try:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(feats)
        except Exception:
            continue
        clusters_data = [
            {'visual_word_index': i, 'centroid': km.cluster_centers_[i],
             'patch_locations': []}
            for i in range(k)
        ]
        for idx, label in enumerate(km.labels_):
            clusters_data[label]['patch_locations'].append(metas[idx])
        page_data[img_name] = {'vocabulary': km.cluster_centers_,
                                'clusters_data': clusters_data}
    return page_data


def _eval_page_data(page_data: dict, method_name: str) -> dict:
    """Build Hungarian-L2 matrix and evaluate. Shared ablation helper."""
    page_names = sorted(page_data.keys())
    n  = len(page_names)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hungarian_l2(page_data[page_names[i]]['vocabulary'],
                             page_data[page_names[j]]['vocabulary'])
            dm[i, j] = dm[j, i] = d
    return compute_retrieval_metrics(dm, page_names, method_name=method_name)


def run_ablation_k(features_cache: dict,
                   k_values: tuple = ABLATION_K_VALUES) -> pd.DataFrame:
    """Ablation: vocabulary size k. Reuses cached features, only re-clusters."""
    print(f"\n{'='*55}\nABLATION — Vocabulary Size k ∈ {k_values}\n{'='*55}")
    results = [_eval_page_data(_cluster_cache(features_cache, k), f'BoB k={k}')
               for k in k_values]
    df = pd.DataFrame(results).set_index('method')
    print("\n" + df[[c for c in ABLATION_DISPLAY_COLS if c in df.columns]].to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_k.csv'))
    return df


def plot_distance_distributions(bob_dm: np.ndarray, bob_names: list,
                                bow_dm: np.ndarray, bow_names: list,
                                label_fn=get_cluster_label,
                                n_boot: int = 1000,
                                bob_chamfer_dm: np.ndarray = None):
    """
    Side-by-side intra vs inter-cluster distance histograms for BoB and BoW.
    Also computes within-method separation statistics and saves them to CSV.
    """
    stats_rows = []

    entries = [
        (bob_dm,  bob_names, 'BoB (Hungarian-L2)'),
        (bow_dm,  bow_names, 'BoW (Cosine)'),
    ]
    if bob_chamfer_dm is not None:
        entries.append((bob_chamfer_dm, bob_names, 'BoB (Chamfer)'))

    ncols = len(entries)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4))
    if ncols == 1:
        axes = [axes]  # ensure always iterable

    for ax, (dm, names, title) in zip(axes, entries):
        labels = np.array([label_fn(n) for n in names])

        idx_i, idx_j = np.triu_indices(len(names), k=1)
        same_cluster  = labels[idx_i] == labels[idx_j]
        vals          = dm[idx_i, idx_j]
        intra         = vals[same_cluster]
        inter         = vals[~same_cluster]

        ax.hist(inter, bins=60, alpha=0.6, color='tomato',    density=True, label='Inter-cluster')
        ax.hist(intra, bins=60, alpha=0.8, color='steelblue', density=True, label='Intra-cluster')

        stats = compute_distance_separation_stats(
            dm, names, label_fn=label_fn, method_name=title, n_boot=n_boot
        )
        stats_rows.append(stats)

        ax.text(
            0.98, 0.97,
            f"KS={stats['ks_stat']:.3f}\nAUC={stats['auc_sep']:.3f}\nd={stats['cohen_d']:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6)
        )
        ax.set_title(f'Distance Distribution\n{title}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_distance_distribution.png'), dpi=150)
    plt.show()

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'distance_separation_stats.csv'), index=False)

    print("\n===== DISTANCE SEPARATION STATS =====")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return stats_df


def _retrain(train_dir: str, encoded_dim: int,
             sparsity_weight: float, save_path: str):
    """Train a fresh autoencoder with given dim and sparsity."""
    if os.path.exists(save_path):
        return
    dataset   = TextPatchDataset(train_dir, PATCH_SIZE, PATCHES_PER_IMAGE_TRAIN,
                                  transform=transforms.ToTensor())
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
    model_abl = SparseAutoencoder(encoded_dim).to(DEVICE)
    opt       = optim.Adam(model_abl.parameters(), lr=LEARNING_RATE)
    crit      = nn.MSELoss()
    model_abl.train()
    for epoch in range(EPOCHS):
        for patches in loader:
            patches = patches.to(DEVICE)
            enc, dec = model_abl(patches)
            loss = crit(dec, patches) + sparsity_weight * torch.sum(torch.abs(enc)) / enc.size(0)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  dim={encoded_dim} λ={sparsity_weight} ep {epoch+1}/{EPOCHS}")
    torch.save(model_abl.state_dict(), save_path)


def run_ablation_encoded_dim(train_dir: str, feat_dir: str,
                              dim_values: tuple = ABLATION_DIM_VALUES) -> pd.DataFrame:
    """Ablation: encoded dimension. Retrains autoencoder for each dim."""
    print(f"\n{'='*55}\nABLATION — Encoded Dim ∈ {dim_values}\n{'='*55}")
    results = []
    for dim in dim_values:
        mp = os.path.join(OUTPUT_DIR, f'model_dim{dim}.pth')
        _retrain(train_dir, dim, SPARSITY_WEIGHT, mp)
        m = SparseAutoencoder(dim).to(DEVICE)
        m.load_state_dict(torch.load(mp, map_location=DEVICE)); m.eval()
        cache = extract_all_features(m, feat_dir,
                    cache_path=os.path.join(OUTPUT_DIR, f'features_dim{dim}.pkl'))
        results.append(_eval_page_data(_cluster_cache(cache, VOCAB_SIZE), f'dim={dim}'))
    df = pd.DataFrame(results).set_index('method')
    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_dim.csv'))
    print("\n" + df[[c for c in ABLATION_DISPLAY_COLS if c in df.columns]].to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def run_ablation_sparsity(train_dir: str, feat_dir: str) -> pd.DataFrame:
    """Ablation: sparsity penalty ON (λ=1e-5) vs OFF (λ=0)."""
    print(f"\n{'='*55}\nABLATION — Sparsity ON vs OFF\n{'='*55}")
    results = []
    for lam, tag in [(SPARSITY_WEIGHT, 'ON'), (0.0, 'OFF')]:
        mp = os.path.join(OUTPUT_DIR, f'model_sparsity_{tag}.pth')
        _retrain(train_dir, ENCODED_DIM, lam, mp)
        m = SparseAutoencoder(ENCODED_DIM).to(DEVICE)
        m.load_state_dict(torch.load(mp, map_location=DEVICE)); m.eval()
        cache = extract_all_features(m, feat_dir,
                    cache_path=os.path.join(OUTPUT_DIR, f'features_sparsity_{tag}.pkl'))
        results.append(_eval_page_data(_cluster_cache(cache, VOCAB_SIZE),
                                        f'Sparsity {tag}'))
    df = pd.DataFrame(results).set_index('method')
    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_sparsity.csv'))
    print("\n" + df[[c for c in ABLATION_DISPLAY_COLS if c in df.columns]].to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def run_ablation_fitpad(model: SparseAutoencoder, feat_dir: str) -> pd.DataFrame:
    """
    Ablation: fit-and-pad ON vs OFF.
    OFF = direct resize to PATCH_SIZE×PATCH_SIZE (no aspect-ratio preserving pad).
    Characters appear stretched but no black padding around them.
    """
    import cv2
    print(f"\n{'='*55}\nABLATION — Fit-and-Pad ON vs OFF\n{'='*55}")

    def _extract_no_fitpad(img_path):
        try:
            from PIL import Image as PILImage
            img_array = np.array(PILImage.open(img_path).convert('L'))
            if np.mean(img_array) > 128:
                img_array = 255 - img_array
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                img_array, connectivity=8)
            patches, metas = [], []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if not (MIN_AREA <= area <= MAX_AREA):
                    continue
                x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
                # DIRECT resize — no fit-and-pad
                crop    = img_array[y:y+h, x:x+w]
                resized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE),
                                     interpolation=cv2.INTER_AREA)
                if np.mean(resized > 128) < MIN_CANVAS_WHITE_RATIO:
                    continue
                patches.append(resized)
                metas.append({'img_path': img_path,
                               'coords': (int(centroids[i, 1]), int(centroids[i, 0]))})
            if len(patches) < MIN_VALID_COMPONENTS:
                return None, None
            return _encode_patches(model, patches), metas
        except Exception:
            return None, None

    image_paths = (glob.glob(os.path.join(feat_dir, '**', '*.png'), recursive=True) +
                   glob.glob(os.path.join(feat_dir, '**', '*.jpg'), recursive=True))

    cache_off = {}
    model.eval()
    for img_path in tqdm(image_paths, desc="Extracting (fit-pad OFF)"):
        feats, metas = _extract_no_fitpad(img_path)
        if feats is not None:
            cache_off[os.path.basename(img_path)] = (feats, metas)

    cache_on = extract_all_features(model, feat_dir)

    results = []
    for tag, cache in [('ON', cache_on), ('OFF', cache_off)]:
        results.append(_eval_page_data(_cluster_cache(cache, VOCAB_SIZE),
                                        f'Fit-Pad {tag}'))
    df = pd.DataFrame(results).set_index('method')
    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_fitpad.csv'))
    print("\n" + df[[c for c in ABLATION_DISPLAY_COLS if c in df.columns]].to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ==============================================================================
# SECTION 13: RUNTIME PROFILING TABLE
# ==============================================================================

def run_runtime_profiling(page_data: dict, bow_data: dict,
                          bob_dm: np.ndarray, bob_names: list,
                          bob_ot_dm: np.ndarray = None) -> pd.DataFrame:
    """Wall-clock time and memory per retrieval strategy."""
    bow_names = bow_data['page_names']
    common    = sorted(set(bob_names) & set(bow_names))
    bi        = [bob_names.index(n) for n in common]
    bwi       = [bow_names.index(n)  for n in common]

    bob_sub = bob_dm[np.ix_(bi, bi)]
    bow_cos = bow_data['dm_cosine'][np.ix_(bwi, bwi)]
    n       = len(common)
    rows    = []

    def _time_lookup(dm):
        t0 = time.perf_counter()
        for i in range(n):
            np.argsort(dm[i])[:10]
        return (time.perf_counter() - t0) / n * 1000  # ms/query

    def _mem_mb(arr):
        return arr.nbytes / 1e6

    # BoW matrix lookup
    rows.append({
        'Method': 'BoW-Cosine (matrix lookup)',
        'ms/query': f'{_time_lookup(bow_cos):.3f}',
        'Full gallery (s)': f'{_time_lookup(bow_cos) * n / 1000:.2f}',
        'Memory (MB)': f'{_mem_mb(bow_cos):.1f}'
    })

    # BoB Hungarian matrix lookup
    rows.append({
        'Method': 'BoB-Hungarian (matrix lookup)',
        'ms/query': f'{_time_lookup(bob_sub):.3f}',
        'Full gallery (s)': f'{_time_lookup(bob_sub) * n / 1000:.2f}',
        'Memory (MB)': f'{_mem_mb(bob_sub):.1f}'
    })

    # BoB Hungarian on-the-fly
    sample_names = random.sample(common, min(20, n))
    t0 = time.perf_counter()
    for i in range(len(sample_names)):
        for j in range(i + 1, len(sample_names)):
            hungarian_l2(
                page_data[sample_names[i]]['vocabulary'],
                page_data[sample_names[j]]['vocabulary']
            )
    elapsed   = time.perf_counter() - t0
    n_pairs   = len(sample_names) * (len(sample_names) - 1) / 2
    ms_pair   = elapsed / n_pairs * 1000
    full_time = ms_pair * n * (n - 1) / 2 / 1000
    rows.append({
        'Method': 'BoB-Hungarian (on-the-fly per pair)',
        'ms/query': f'{ms_pair * n:.2f}',
        'Full gallery (s)': f'{full_time:.1f}',
        'Memory (MB)': '0.0 (no matrix)'
    })

    # Optional OT matrix lookup
    if bob_ot_dm is not None:
        ot_sub = bob_ot_dm[np.ix_(bi, bi)]
        rows.append({
            'Method': 'BoB-OT (matrix lookup)',
            'ms/query': f'{_time_lookup(ot_sub):.3f}',
            'Full gallery (s)': f'{_time_lookup(ot_sub) * n / 1000:.2f}',
            'Memory (MB)': f'{_mem_mb(ot_sub):.1f}'
        })

    # Two-stage rerank: BoW lookup + BoB-OT on top-M
    t0 = time.perf_counter()
    for i in range(n):
        dists = bow_cos[i].copy()
        dists[i] = np.inf
        top_m = np.argsort(dists)[:RERANK_TOP_M]

        for j in top_m:
            bob_ot_weighted(
                page_data[common[i]]['vocabulary'],
                page_data[common[i]]['clusters_data'],
                page_data[common[j]]['vocabulary'],
                page_data[common[j]]['clusters_data'],
            )

    ms_rerank = (time.perf_counter() - t0) / n * 1000
    rows.append({
        'Method': f'BoW-Cosine + BoB-OT rerank (top-{RERANK_TOP_M})',
        'ms/query': f'{ms_rerank:.2f}',
        'Full gallery (s)': f'{ms_rerank * n / 1000:.2f}',
        'Memory (MB)': f'{_mem_mb(bow_cos):.1f} (BoW only)'
    })

    df = pd.DataFrame(rows).set_index('Method')
    print("\n===== RUNTIME PROFILING =====")
    print(df.to_string())
    df.to_csv(os.path.join(OUTPUT_DIR, 'runtime_profiling.csv'))
    return df


def _load_or_train_model_for_patch_mode(train_dir: str, train_mode: str) -> SparseAutoencoder:
    from .dataset import train_autoencoder
    model_path = os.path.join(OUTPUT_DIR, f'bob_model_{train_mode}.pth')
    recon_path = os.path.join(OUTPUT_DIR, f'fig_reconstructions_{train_mode}.png')
    manifest_path = os.path.join(OUTPUT_DIR, f'train_manifest_{train_mode}.csv')

    model = SparseAutoencoder(ENCODED_DIM).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model

    return train_autoencoder(
        image_dir=train_dir,
        patch_mode=train_mode,
        model_save_path=model_path,
        recon_save_path=recon_path,
        manifest_save_path=manifest_path,
    )


def _evaluate_bob_hungarian(page_data: dict, method_name: str, label_fn=get_cluster_label) -> dict:
    page_names = sorted(page_data.keys())
    n = len(page_names)
    dm = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = hungarian_l2(
                page_data[page_names[i]]['vocabulary'],
                page_data[page_names[j]]['vocabulary']
            )
            dm[i, j] = dm[j, i] = d

    results = compute_retrieval_metrics(dm, page_names, label_fn=label_fn, method_name=method_name)
    results['NumPages'] = len(page_names)

    labels = [label_fn(n) for n in page_names]
    num_queries = 0
    for i, gt in enumerate(labels):
        positives = [j for j in range(len(labels)) if j != i and labels[j] == gt]
        if positives:
            num_queries += 1
    results['NumQueries'] = num_queries
    return results


def run_patch_mode_ablation(
    train_dir: str,
    feat_dir: str,
    patch_mode_pairs: List[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Main experiment:
      - train=context, infer=fitpad
      - train=fitpad, infer=fitpad
    """
    if patch_mode_pairs is None:
        patch_mode_pairs = PATCH_MODE_ABLATIONS

    all_results = []

    for train_mode, infer_mode in patch_mode_pairs:
        print("\n" + "=" * 70)
        print(f"PATCH MODE EXPERIMENT: train={train_mode}, infer={infer_mode}")
        print("=" * 70)

        model = _load_or_train_model_for_patch_mode(train_dir, train_mode)

        save_path = os.path.join(OUTPUT_DIR, f'bob_page_data_train-{train_mode}_infer-{infer_mode}.pkl')
        manifest_csv = os.path.join(OUTPUT_DIR, f'page_filtering_log_train-{train_mode}_infer-{infer_mode}.csv')

        page_data = build_bob_vocabularies(
            model=model,
            image_dir=feat_dir,
            patch_mode=infer_mode,
            save_path=save_path,
            manifest_csv=manifest_csv,
        )

        result = _evaluate_bob_hungarian(
            page_data=page_data,
            method_name=f'BoB-Hungarian train={train_mode} infer={infer_mode}'
        )
        result['TrainPatchMode'] = train_mode
        result['InferPatchMode'] = infer_mode
        all_results.append(result)

    df = pd.DataFrame(all_results)
    cols = [
        'TrainPatchMode', 'InferPatchMode',
        'Hit@1', 'Hit@5', 'mAP@5', 'MRR', 'MacroF1@1',
        'NumPages', 'NumQueries'
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out_csv = os.path.join(OUTPUT_DIR, 'ablation_patch_mode.csv')
    df.to_csv(out_csv, index=False)

    print("\n===== PATCH MODE ABLATION =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print(f"\nSaved to {out_csv}")
    return df
