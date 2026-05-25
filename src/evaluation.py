# -*- coding: utf-8 -*-
"""
RETRIEVAL EVALUATION
Retrieval metrics (Hit@k, mAP@k, MRR, MacroF1@1) and the full evaluation table.
"""

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from .config import EVAL_KS, RERANK_TOP_M, RESULTS_CSV
from .distances import get_cluster_label, bob_ot_weighted
from .baselines import _reorder_dm_to_expected_names


def compute_retrieval_metrics(distance_matrix: np.ndarray, page_names: list,
                               label_fn=get_cluster_label,
                               ks: tuple = EVAL_KS,
                               method_name: str = 'method') -> dict:
    """
    Compute Hit@k, mAP@k, MRR, MacroF1@1 from a symmetric pairwise distance matrix.
    Queries with no matching positives in the dataset are skipped.
    """
    labels = [label_fn(n) for n in page_names]
    N      = len(page_names)
    hit    = {k: [] for k in ks}
    ap     = {k: [] for k in ks}
    rr     = []
    preds, gts = [], []

    for i in range(N):
        gt    = labels[i]
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        ranked = np.argsort(dists)

        positives = [j for j in range(N) if j != i and labels[j] == gt]
        if not positives:
            continue

        total_pos = len(positives)

        # Reciprocal rank over the FULL ranking
        first_rel_rank = None
        for rank, j in enumerate(ranked, 1):
            if labels[j] == gt:
                first_rel_rank = rank
                break
        rr.append(1.0 / first_rel_rank if first_rel_rank is not None else 0.0)

        for k in ks:
            top_k      = ranked[:k]
            top_labels = [labels[j] for j in top_k]
            hit[k].append(int(gt in top_labels))

            num_rel, _ap = 0, 0.0
            for rank, j in enumerate(top_k, 1):
                if labels[j] == gt:
                    num_rel += 1
                    _ap     += num_rel / rank
            ap[k].append(_ap / min(total_pos, k) if min(total_pos, k) > 0 else 0.0)

        preds.append(labels[ranked[0]])
        gts.append(gt)

    macro_f1 = f1_score(gts, preds, average='macro', zero_division=0) if gts else 0.0

    results = {'method': method_name}
    for k in ks:
        results[f'Hit@{k}'] = float(np.mean(hit[k])) if hit[k] else 0.0
        results[f'mAP@{k}'] = float(np.mean(ap[k])) if ap[k] else 0.0
    results['MRR']       = float(np.mean(rr)) if rr else 0.0
    results['MacroF1@1'] = float(macro_f1)

    print(f"\n{'='*52}\n  {method_name}\n{'='*52}")
    for k in ks:
        print(f"  Hit@{k:<3} = {results[f'Hit@{k}']:.4f}   "
              f"mAP@{k:<3} = {results[f'mAP@{k}']:.4f}")
    print(f"  MRR       = {results['MRR']:.4f}")
    print(f"  MacroF1@1 = {results['MacroF1@1']:.4f}\n{'='*52}")
    return results


def _build_reranked_matrix(bow_dm: np.ndarray, page_names: list,
                            page_data_sub: dict) -> np.ndarray:
    """BoW top-M retrieval then rescore with BoB-OT weighted distance."""
    N        = len(page_names)
    reranked = np.full((N, N), np.inf)
    np.fill_diagonal(reranked, 0.0)
    for i in tqdm(range(N), desc="Re-ranking"):
        dists    = bow_dm[i].copy()
        dists[i] = np.inf
        top_m    = np.argsort(dists)[:RERANK_TOP_M]
        for j in top_m:
            d = bob_ot_weighted(
                page_data_sub[page_names[i]]['vocabulary'],
                page_data_sub[page_names[i]]['clusters_data'],
                page_data_sub[page_names[j]]['vocabulary'],
                page_data_sub[page_names[j]]['clusters_data'],
            )
            reranked[i, j] = d
    reranked = np.minimum(reranked, reranked.T)
    return reranked


def _append_flat_pool_rows(results, flat_data, expected_page_names, label_fn):
    """
    Add Page-MeanPool / Page-MaxPool rows with strict page alignment.
    """
    fp_names = list(flat_data['page_names'])

    row_map = [
        ('dm_mean_l2', 'Page-MeanPool-L2'),
        ('dm_mean_cosine', 'Page-MeanPool-Cosine'),
        ('dm_max_l2', 'Page-MaxPool-L2'),
        ('dm_max_cosine', 'Page-MaxPool-Cosine'),
    ]

    for dm_key, method_name in row_map:
        if dm_key not in flat_data:
            continue

        dm = _reorder_dm_to_expected_names(
            dm=flat_data[dm_key],
            source_names=fp_names,
            expected_names=expected_page_names,
            prefix=method_name,
        )

        results.append(compute_retrieval_metrics(
            dm,
            expected_page_names,
            label_fn=label_fn,
            method_name=method_name,
        ))


def _append_bow_rows(results, bow_data, expected_page_names, prefix, label_fn):
    bow_page_names = list(bow_data['page_names'])

    if set(bow_page_names) != set(expected_page_names):
        missing_in_bow = sorted(set(expected_page_names) - set(bow_page_names))
        extra_in_bow   = sorted(set(bow_page_names) - set(expected_page_names))
        raise ValueError(
            f"{prefix}: page-set mismatch.\n"
            f"Missing in BoW: {len(missing_in_bow)}\n"
            f"Extra in BoW: {len(extra_in_bow)}"
        )

    # Reorder BoW matrices to match expected_page_names exactly
    idx_map = {name: i for i, name in enumerate(bow_page_names)}
    idx = [idx_map[name] for name in expected_page_names]

    metric_map = [
        ('dm_l2', 'L2'),
        ('dm_cosine', 'Cosine'),
        ('dm_chi2', 'Chi2'),
        ('dm_hellinger', 'Hellinger'),
    ]

    for dm_key, suffix in metric_map:
        if dm_key not in bow_data:
            continue

        dm = bow_data[dm_key][np.ix_(idx, idx)]

        results.append(compute_retrieval_metrics(
            dm,
            expected_page_names,
            label_fn=label_fn,
            method_name=f'{prefix}-{suffix}',
        ))


def run_full_evaluation(
    bob_dm:             np.ndarray,
    bob_names:          list,
    bow_centroids_data: dict,
    page_data:          dict,
    bow_rawpatch_data:  Optional[dict]       = None,
    flat_pool_data:     Optional[dict]       = None,
    bob_chamfer_dm:     Optional[np.ndarray] = None,
    bob_ot_dm:          Optional[np.ndarray] = None,
    label_fn                                 = get_cluster_label,
) -> pd.DataFrame:
    """
    Master evaluation table covering:
      - BoW-Centroids-{L2, Cosine, Chi2, Hellinger}
      - BoW-RawPatches-{L2, Cosine, Chi2, Hellinger}
      - Page-MeanPool-{L2, Cosine}
      - Page-MaxPool-{L2, Cosine}  [optional]
      - BoB-Hungarian
      - BoB-Chamfer
      - BoB-OT
      - BoW-Centroids-Cosine + BoB-OT rerank
    All methods are evaluated on the SAME page set: bob_names.
    """
    results = []

    # --- BoW rows ---
    _append_bow_rows(
        results=results,
        bow_data=bow_centroids_data,
        expected_page_names=bob_names,
        prefix='BoW-Centroids',
        label_fn=label_fn,
    )

    if bow_rawpatch_data is not None:
        _append_bow_rows(
            results=results,
            bow_data=bow_rawpatch_data,
            expected_page_names=bob_names,
            prefix='BoW-RawPatches',
            label_fn=label_fn,
        )

    # --- Flat pooled page baselines ---
    if flat_pool_data is not None:
        _append_flat_pool_rows(
            results=results,
            flat_data=flat_pool_data,
            expected_page_names=bob_names,
            label_fn=label_fn,
        )

    # --- BoB rows ---
    if bob_chamfer_dm is not None:
        results.append(compute_retrieval_metrics(
            bob_chamfer_dm,
            bob_names,
            label_fn=label_fn,
            method_name='BoB-Chamfer'
        ))

    results.append(compute_retrieval_metrics(
        bob_dm,
        bob_names,
        label_fn=label_fn,
        method_name='BoB-Hungarian'
    ))

    if bob_ot_dm is not None:
        results.append(compute_retrieval_metrics(
            bob_ot_dm,
            bob_names,
            label_fn=label_fn,
            method_name='BoB-OT'
        ))

        # two-stage reranking using BoW-Centroids cosine as shortlist
        bow_cos = _reorder_dm_to_expected_names(
            dm=bow_centroids_data['dm_cosine'],
            source_names=list(bow_centroids_data['page_names']),
            expected_names=bob_names,
            prefix='BoW-Centroids-Cosine rerank source',
        )

        reranked = _build_reranked_matrix(
            bow_cos,
            bob_names,
            {n: page_data[n] for n in bob_names}
        )

        results.append(compute_retrieval_metrics(
            reranked,
            bob_names,
            label_fn=label_fn,
            method_name=f'BoW-Centroids-Cosine + BoB-OT rerank (top-{RERANK_TOP_M})'
        ))

    # --- print + save ---
    cols = [
        'method',
        'Hit@1', 'mAP@1',
        'Hit@5', 'mAP@5',
        'Hit@10', 'mAP@10',
        'MRR', 'MacroF1@1'
    ]
    df = pd.DataFrame(results)
    df = df[[c for c in cols if c in df.columns]]

    if 'method' in df.columns:
        df = df.sort_values('method').reset_index(drop=True)

    print("\n===== FULL RETRIEVAL COMPARISON =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved → {RESULTS_CSV}")

    return df
