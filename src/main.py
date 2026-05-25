# -*- coding: utf-8 -*-
"""
MAIN ENTRY POINT
Bag-of-Bags (BoB) for Genizah Manuscript Join Retrieval — full pipeline.

Pipeline:
  1. Train a sparse convolutional autoencoder on character patches
  2. Build per-image BoB vocabulary via KMeans on encoded patches
  3. Build BoW baseline (TF-IDF histogram over global codebook)
  4. Compute pairwise distance matrices (Hungarian-L2, Chamfer, Hungarian-Cosine, BoW variants)
  5. Evaluate retrieval: Hit@k, mAP@k, MacroF1@1
  6. Visualize: reconstruction grid, distance distributions, PCA scatter, heatmap
  7. BCC graph analysis for join discovery

Run from the project root:
    python -m src.main
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from .config import (
    DEVICE, ENCODED_DIM, TRAIN_PATCH_MODE, INFER_PATCH_MODE,
    TRAIN_IMAGE_DIR, FEAT_IMAGE_DIR, OUTPUT_DIR,
    BOW_VOCAB_SIZE, BOW_CENTROIDS_DATA_FILE, BOW_RAWPATCH_DATA_FILE,
    FLAT_POOL_FILE, SIMILARITY_THRESHOLD, RUN_MAXPOOL,
)
from .model import SparseAutoencoder
from .dataset import train_autoencoder
from .features import (
    build_bob_vocabularies, make_folder_label_fn, validate_label_consistency,
    extract_all_features,
)
from .baselines import (
    build_bow_centroids_representation, build_bow_rawpatch_representation,
    build_flat_pool_baselines,
)
from .distances import build_bob_distance_matrix, build_bob_ot_matrix
from .evaluation import run_full_evaluation
from .visualization import (
    visualize_matched_pairs,
)
from .bcc import run_bcc_analysis, plot_query_topk_grid
from .ablations import (
    run_ablation_k, run_ablation_encoded_dim, run_ablation_sparsity,
    plot_distance_distributions, run_runtime_profiling,
)


if __name__ == '__main__':

    RUN_CHAMFER = True
    RUN_OT = False

    main_model_path = os.path.join(OUTPUT_DIR, f'bob_model_{TRAIN_PATCH_MODE}.pth')
    main_bob_path   = os.path.join(OUTPUT_DIR, f'bob_page_data_{INFER_PATCH_MODE}.pkl')

    # Train/load model
    if not os.path.exists(main_model_path):
        model = train_autoencoder(
            TRAIN_IMAGE_DIR,
            patch_mode=TRAIN_PATCH_MODE,
            model_save_path=main_model_path,
            recon_save_path=os.path.join(OUTPUT_DIR, f'fig_reconstructions_{TRAIN_PATCH_MODE}.png'),
            manifest_save_path=os.path.join(OUTPUT_DIR, f'train_manifest_{TRAIN_PATCH_MODE}.csv'),
        )
    else:
        print(f"Loading model from {main_model_path}")
        model = SparseAutoencoder(ENCODED_DIM).to(DEVICE)
        model.load_state_dict(torch.load(main_model_path, map_location=DEVICE))
        model.eval()

    # Build/load BoB page data
    if not os.path.exists(main_bob_path):
        page_data = build_bob_vocabularies(
            model,
            FEAT_IMAGE_DIR,
            patch_mode=INFER_PATCH_MODE,
            save_path=main_bob_path,
        )
    else:
        print(f"Loading BoB data from {main_bob_path}")
        with open(main_bob_path, 'rb') as f:
            page_data = pickle.load(f)
        print(f"  {len(page_data)} pages loaded.")

    if len(page_data) == 0:
        raise RuntimeError("No pages survived filtering. Check thresholds and patch extraction.")

    # Labels
    validate_label_consistency(page_data)
    folder_label_fn = make_folder_label_fn(page_data)

    # Distance matrices
    bob_dm, bob_names = build_bob_distance_matrix(page_data, metric='hungarian_l2')
    df_bob = pd.DataFrame(bob_dm, index=bob_names, columns=bob_names)

    bob_chamfer_dm = None
    if RUN_CHAMFER:
        bob_chamfer_dm, _ = build_bob_distance_matrix(page_data, metric='chamfer')

    bob_ot_dm = None
    if RUN_OT:
        bob_ot_dm, _ = build_bob_ot_matrix(page_data)

    # Cache raw features once (same encoder, same infer mode)
    features_cache = extract_all_features(
        model,
        FEAT_IMAGE_DIR,
        patch_mode=INFER_PATCH_MODE,
    )

    eligible_page_names = sorted(page_data.keys())   # fairness anchor

    # Build/load BoW-Centroids
    if not os.path.exists(BOW_CENTROIDS_DATA_FILE):
        bow_centroids_data = build_bow_centroids_representation(
            page_data=page_data,
            eligible_page_names=eligible_page_names,
            vocab_size=BOW_VOCAB_SIZE,
            save_path=BOW_CENTROIDS_DATA_FILE,
        )
    else:
        print(f"Loading BoW-Centroids from {BOW_CENTROIDS_DATA_FILE}")
        with open(BOW_CENTROIDS_DATA_FILE, 'rb') as f:
            bow_centroids_data = pickle.load(f)

    # Build/load BoW-RawPatches
    if not os.path.exists(BOW_RAWPATCH_DATA_FILE):
        bow_rawpatch_data = build_bow_rawpatch_representation(
            features_cache=features_cache,
            eligible_page_names=eligible_page_names,
            vocab_size=BOW_VOCAB_SIZE,
            save_path=BOW_RAWPATCH_DATA_FILE,
        )
    else:
        print(f"Loading BoW-RawPatches from {BOW_RAWPATCH_DATA_FILE}")
        with open(BOW_RAWPATCH_DATA_FILE, 'rb') as f:
            bow_rawpatch_data = pickle.load(f)

    # Flat pooling baselines
    _pool_modes = ('mean', 'max') if RUN_MAXPOOL else ('mean',)

    if not os.path.exists(FLAT_POOL_FILE):
        flat_pool_data = build_flat_pool_baselines(
            features_cache=features_cache,
            eligible_page_names=eligible_page_names,
            pool_modes=_pool_modes,
            save_path=FLAT_POOL_FILE,
        )
    else:
        print(f"Loading flat-pool baselines from {FLAT_POOL_FILE}")
        with open(FLAT_POOL_FILE, 'rb') as f:
            flat_pool_data = pickle.load(f)

    # Full evaluation table
    df_results = run_full_evaluation(
        bob_dm             = bob_dm,
        bob_names          = bob_names,
        bow_centroids_data = bow_centroids_data,
        page_data          = page_data,
        bow_rawpatch_data  = bow_rawpatch_data,
        flat_pool_data     = flat_pool_data,
        bob_chamfer_dm     = bob_chamfer_dm,
        bob_ot_dm          = bob_ot_dm,
        label_fn           = folder_label_fn,
    )

    # Diagnostics
    bow_plot_data = bow_rawpatch_data   # or bow_centroids_data if you prefer

    common = sorted(set(bob_names) & set(bow_plot_data['page_names']))
    bi = [bob_names.index(n) for n in common]
    bwi = [bow_plot_data['page_names'].index(n) for n in common]

    if len(common) > 0:
        stats_df = plot_distance_distributions(
            bob_dm[np.ix_(bi, bi)], common,
            bow_plot_data['dm_cosine'][np.ix_(bwi, bwi)], common,
            label_fn=folder_label_fn, n_boot=200,
            bob_chamfer_dm=bob_chamfer_dm[np.ix_(bi, bi)] if bob_chamfer_dm is not None else None
        )

    # Runtime
    run_runtime_profiling(
        page_data,
        bow_centroids_data,   # keep runtime aligned with reranking source
        bob_dm,
        bob_names,
        bob_ot_dm=bob_ot_dm
    )

    # Ablations
    run_ablation_k(features_cache)
    run_ablation_encoded_dim(TRAIN_IMAGE_DIR, FEAT_IMAGE_DIR)
    run_ablation_sparsity(TRAIN_IMAGE_DIR, FEAT_IMAGE_DIR)

    # run_ablation_fitpad(model, FEAT_IMAGE_DIR)

    # BCC
    run_bcc_analysis(df_bob, page_data, threshold=SIMILARITY_THRESHOLD)

    # Qualitative checks
    if len(bob_names) >= 1:
        QUERY_PAGE = bob_names[0]
        print(f"\nTop 20 nearest to '{QUERY_PAGE}':")
        print(df_bob[QUERY_PAGE].sort_values().iloc[1:21].to_string())

    if len(bob_names) >= 2:
        visualize_matched_pairs(bob_names[0], bob_names[1], page_data, df_bob, n_samples=5)

    if len(common) >= 1:
        plot_query_topk_grid(
            common[0],
            bob_dm[np.ix_(bi, bi)],
            common,
            {n: page_data[n] for n in common},
            label_fn=folder_label_fn,
            k=5,
            save_name='fig_query_top5_grid.png'
        )

