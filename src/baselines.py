# -*- coding: utf-8 -*-
"""
FLAT-POOLING BASELINES AND BOW
Includes BoW-Centroids, BoW-RawPatches, and page-level pooling baselines.
"""

import os
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .config import (
    SEED, OUTPUT_DIR, INFER_PATCH_MODE, BOW_VOCAB_SIZE,
    FLAT_POOL_FILE,
)


# ==============================================================================
# SIMPLE FLAT-POOLING BASELINES
# ==============================================================================

def _cache_get_features(cache_item):
    """
    Supports BOTH cache formats:
      1) tuple: (feats, metas)
      2) dict: {'features': feats, 'metas' / 'metadata': ...}
    Returns feats as np.ndarray or None.
    """
    if isinstance(cache_item, tuple):
        if len(cache_item) == 0:
            return None
        return cache_item[0]

    if isinstance(cache_item, dict):
        if 'features' in cache_item:
            return cache_item['features']
        return None

    return None


def _reorder_dm_to_expected_names(dm: np.ndarray, source_names: list, expected_names: list, prefix: str) -> np.ndarray:
    """
    Strict reorder helper. Raises if page sets differ.
    """
    if set(source_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(source_names))
        extra   = sorted(set(source_names) - set(expected_names))
        raise ValueError(
            f"{prefix}: page-set mismatch.\n"
            f"Missing: {len(missing)}\n"
            f"Extra: {len(extra)}"
        )

    idx_map = {name: i for i, name in enumerate(source_names)}
    idx = [idx_map[name] for name in expected_names]
    return dm[np.ix_(idx, idx)]


def build_flat_pool_baselines(
    features_cache: dict,
    eligible_page_names: list,
    save_path: Optional[str] = None,
    pool_modes: tuple = ('mean', 'max'),
) -> dict:
    """
    Build simple page-level baselines from raw component embeddings.

    Uses the SAME features_cache as BoW-RawPatches, so:
      - same encoder
      - same inference patch mode
      - same eligible page set as BoB

    pool_modes: subset of ('mean', 'max')
    Returns:
        {
          'page_names': [...],
          'dm_mean_l2': ...,
          'dm_mean_cosine': ...,
          'dm_max_l2': ...,
          'dm_max_cosine': ...
        }
    """
    if save_path is None:
        save_path = FLAT_POOL_FILE

    supported = {'mean', 'max'}
    if not set(pool_modes).issubset(supported):
        raise ValueError(f"Unsupported pool_modes={pool_modes}. Allowed: {supported}")

    page_names = []
    mean_vecs = []
    max_vecs = []

    missing_pages = []
    empty_pages = []

    for page_id in eligible_page_names:
        if page_id not in features_cache:
            missing_pages.append(page_id)
            continue

        feats = _cache_get_features(features_cache[page_id])
        if feats is None:
            empty_pages.append(page_id)
            continue

        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] == 0:
            empty_pages.append(page_id)
            continue

        page_names.append(page_id)

        if 'mean' in pool_modes:
            mean_vecs.append(feats.mean(axis=0))
        if 'max' in pool_modes:
            max_vecs.append(feats.max(axis=0))

    # Strict fairness: the flat baseline must cover exactly the same pages as BoB
    if missing_pages or empty_pages or set(page_names) != set(eligible_page_names):
        missing = sorted(set(eligible_page_names) - set(page_names))
        extra   = sorted(set(page_names) - set(eligible_page_names))
        raise ValueError(
            f"Flat-pool page-set mismatch.\n"
            f"Missing pages: {len(missing)}\n"
            f"Extra pages: {len(extra)}\n"
            f"Missing-from-cache: {len(missing_pages)}\n"
            f"Empty/invalid: {len(empty_pages)}"
        )

    if len(page_names) == 0:
        raise RuntimeError("build_flat_pool_baselines: no eligible pages found.")

    print(f"\nBuilding flat-pool baselines {pool_modes} on {len(page_names)} pages...")

    flat_data = {'page_names': page_names}

    if 'mean' in pool_modes:
        E_mean = np.stack(mean_vecs, axis=0).astype(np.float32)
        flat_data['dm_mean_l2'] = cdist(E_mean, E_mean, metric='euclidean').astype(np.float32)
        flat_data['dm_mean_cosine'] = cdist(E_mean, E_mean, metric='cosine').astype(np.float32)
        print(f"  MeanPool embedding shape: {E_mean.shape}")

    if 'max' in pool_modes:
        E_max = np.stack(max_vecs, axis=0).astype(np.float32)
        flat_data['dm_max_l2'] = cdist(E_max, E_max, metric='euclidean').astype(np.float32)
        flat_data['dm_max_cosine'] = cdist(E_max, E_max, metric='cosine').astype(np.float32)
        print(f"  MaxPool embedding shape: {E_max.shape}")

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(flat_data, f)
        print(f"  Flat-pool data saved → {save_path}")

    return flat_data


# ==============================================================================
# BOW BASELINE BUILD
# ==============================================================================

def _chi2_matrix(H: np.ndarray) -> np.ndarray:
    n = H.shape[0]
    dm = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            denom = H[i] + H[j]
            mask = denom > 0
            d = np.sum(((H[i][mask] - H[j][mask]) ** 2) / (denom[mask] + 1e-12))
            dm[i, j] = dm[j, i] = float(d)
    return dm


def _hellinger_matrix(H_prob: np.ndarray) -> np.ndarray:
    """
    H_prob must be nonnegative and L1-normalized row-wise.
    Hellinger(P,Q) = ||sqrt(P)-sqrt(Q)||_2 / sqrt(2)
    """
    S = np.sqrt(np.clip(H_prob, 0.0, None))
    return cdist(S, S, metric='euclidean') / np.sqrt(2.0)


def _build_bow_from_page_vectors(
    page_vectors: dict,
    eligible_page_names: list,
    vocab_size: int,
    baseline_name: str,
    save_path: str = None,
) -> dict:
    """
    Shared BoW builder used by BOTH:
      - BoW-Centroids
      - BoW-RawPatches

    page_vectors:
        dict[page_id] -> np.ndarray of shape (n_vectors_for_page, D)

    eligible_page_names:
        must be the SAME page set as BoB (e.g. sorted(page_data.keys()))
    """
    eligible_page_names = [p for p in eligible_page_names if p in page_vectors and len(page_vectors[p]) > 0]
    if len(eligible_page_names) == 0:
        raise RuntimeError(f"{baseline_name}: no eligible pages with vectors.")

    pooled = np.concatenate([page_vectors[p] for p in eligible_page_names], axis=0).astype(np.float32)
    if pooled.shape[0] < vocab_size:
        raise RuntimeError(
            f"{baseline_name}: pooled feature count ({pooled.shape[0]}) < vocab_size ({vocab_size})."
        )

    print(f"\nBuilding {baseline_name}...")
    print(f"  Eligible pages: {len(eligible_page_names)}")
    print(f"  Global feature pool: {pooled.shape}")

    print(f"  Fitting global KMeans (K={vocab_size})...")
    km = KMeans(n_clusters=vocab_size, random_state=SEED, n_init=10)
    km.fit(pooled)
    codebook = km.cluster_centers_.astype(np.float32)

    # --- TF histograms ---
    tf_histograms = {}
    histograms_raw = {}
    histograms_norm = {}
    histograms_prob = {}
    doc_freq = np.zeros(vocab_size, dtype=np.float32)

    for page_id in tqdm(eligible_page_names, desc=f"  Encoding {baseline_name}"):
        vecs = np.asarray(page_vectors[page_id], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            continue

        assignments = np.argmin(cdist(vecs, codebook, metric='euclidean'), axis=1)
        tf = np.bincount(assignments, minlength=vocab_size).astype(np.float32)
        tf /= (tf.sum() + 1e-8)

        tf_histograms[page_id] = tf
        doc_freq += (tf > 0).astype(np.float32)

    page_names = sorted(tf_histograms.keys())
    if len(page_names) == 0:
        raise RuntimeError(f"{baseline_name}: no histograms were built.")

    # --- IDF ---
    N_docs = len(page_names)
    idf = np.log((N_docs + 1) / (doc_freq + 1)) + 1.0

    # --- Build histogram variants ---
    for page_id in page_names:
        h_raw = tf_histograms[page_id] * idf

        l2_norm = np.linalg.norm(h_raw)
        h_norm = h_raw / l2_norm if l2_norm > 0 else h_raw

        l1_norm = h_raw.sum()
        h_prob = h_raw / l1_norm if l1_norm > 0 else h_raw

        histograms_raw[page_id] = h_raw.astype(np.float32)
        histograms_norm[page_id] = h_norm.astype(np.float32)
        histograms_prob[page_id] = h_prob.astype(np.float32)

    H_raw = np.stack([histograms_raw[p] for p in page_names], axis=0)
    H_norm = np.stack([histograms_norm[p] for p in page_names], axis=0)
    H_prob = np.stack([histograms_prob[p] for p in page_names], axis=0)

    print(f"  Computing {baseline_name} distance matrices (L2, cosine, chi2, hellinger)...")
    bow_data = {
        'baseline_name': baseline_name,
        'page_names': page_names,
        'global_codebook': codebook,
        'idf': idf.astype(np.float32),
        'tf_histograms': tf_histograms,
        'histograms_raw': histograms_raw,
        'histograms_norm': histograms_norm,
        'histograms_prob': histograms_prob,
        'dm_l2': cdist(H_raw, H_raw, metric='euclidean').astype(np.float32),
        'dm_cosine': cdist(H_norm, H_norm, metric='cosine').astype(np.float32),
        'dm_chi2': _chi2_matrix(H_prob),
        'dm_hellinger': _hellinger_matrix(H_prob).astype(np.float32),
        'vocab_size': vocab_size,
        'n_pages': len(page_names),
    }

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(bow_data, f)
        print(f"  {baseline_name} saved to {save_path}")

    return bow_data


def build_bow_centroids_representation(
    page_data: dict,
    eligible_page_names: list = None,
    vocab_size: int = BOW_VOCAB_SIZE,
    save_path: str = None,
) -> dict:
    """
    Prototype/global-codebook baseline.
    This is your CURRENT BoW baseline, now renamed clearly to BoW-Centroids.
    """
    if eligible_page_names is None:
        eligible_page_names = sorted(page_data.keys())

    page_vectors = {}
    for page_id in eligible_page_names:
        if page_id not in page_data:
            continue

        entry = page_data[page_id]
        vecs = []
        for cluster in entry['clusters_data']:
            centroid = np.asarray(cluster['centroid'], dtype=np.float32)
            count = max(1, len(cluster['patch_locations']))
            vecs.extend([centroid] * count)

        if len(vecs) > 0:
            page_vectors[page_id] = np.stack(vecs, axis=0)

    return _build_bow_from_page_vectors(
        page_vectors=page_vectors,
        eligible_page_names=eligible_page_names,
        vocab_size=vocab_size,
        baseline_name='BoW-Centroids',
        save_path=save_path,
    )


def build_bow_rawpatch_representation(
    features_cache: dict,
    eligible_page_names: list,
    vocab_size: int = BOW_VOCAB_SIZE,
    save_path: str = None,
) -> dict:
    """
    True classical BoW baseline over RAW component embeddings.

    features_cache:
        dict[page_id] -> (feats, metas)
        where feats has shape (n_components, D)

    eligible_page_names:
        MUST be the SAME evaluated page set as BoB (e.g. sorted(page_data.keys()))
    """
    page_vectors = {}
    missing_pages = []

    for page_id in eligible_page_names:
        if page_id not in features_cache:
            missing_pages.append(page_id)
            continue

        feats, metas = features_cache[page_id]
        if feats is None or len(feats) == 0:
            continue

        page_vectors[page_id] = np.asarray(feats, dtype=np.float32)

    if missing_pages:
        print(f"  Warning: {len(missing_pages)} eligible pages missing from features_cache for BoW-RawPatches.")

    return _build_bow_from_page_vectors(
        page_vectors=page_vectors,
        eligible_page_names=eligible_page_names,
        vocab_size=vocab_size,
        baseline_name='BoW-RawPatches',
        save_path=save_path,
    )
