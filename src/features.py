# -*- coding: utf-8 -*-


import os
import re
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import transforms

from .config import (
    SEED, DEVICE, PATCH_SIZE, VOCAB_SIZE, OUTPUT_DIR, INFER_PATCH_MODE,
)
from .model import SparseAutoencoder
from .dataset import (
    PatchFilterConfig, _list_image_paths, _make_page_id,
    extract_component_records,
)


def _encode_patches(model: SparseAutoencoder, patches: list) -> np.ndarray:
    """Encode a list of HxW numpy patches; returns (N, D) numpy array."""
    to_tensor = transforms.ToTensor()
    import torch
    batch = torch.stack([to_tensor(Image.fromarray(p)) for p in patches]).to(DEVICE)
    with torch.no_grad():
        features = model.encode(batch)
    return features.cpu().numpy()


def get_image_features(
    model: SparseAutoencoder,
    img_path: str,
    patch_mode: str = INFER_PATCH_MODE,
    dataset_root: Optional[str] = None,
    filter_cfg: Optional[PatchFilterConfig] = None,
    return_summary: bool = False,
):
    """
    Shared inference wrapper over extract_component_records.
    """
    filter_cfg = filter_cfg or PatchFilterConfig()

    records, summary = extract_component_records(
        img_path=img_path,
        mode=patch_mode,
        patch_size=PATCH_SIZE,
        filter_cfg=filter_cfg,
        dataset_root=dataset_root,
        patches_per_img=None,  # inference should keep all valid components
    )

    if summary['excluded']:
        return (None, None, summary) if return_summary else (None, None)

    patches = [r['patch'] for r in records]
    metas = [r['meta'] for r in records]
    feats = _encode_patches(model, patches)

    return (feats, metas, summary) if return_summary else (feats, metas)


def build_bob_vocabularies(
    model: SparseAutoencoder,
    image_dir: str,
    patch_mode: str = INFER_PATCH_MODE,
    save_path: Optional[str] = None,
    manifest_csv: Optional[str] = None,
) -> dict:
    """
    Build per-page BoB vocabularies using a shared patch extractor.
    Uses stable page_id keys instead of basenames.
    """
    image_paths = _list_image_paths(image_dir)
    print(f"\nBuilding BoB vocabularies for {len(image_paths)} images... mode={patch_mode}")

    model.eval()
    page_data = {}
    page_manifest = []
    filter_cfg = PatchFilterConfig()

    for img_path in tqdm(image_paths, desc=f"Building vocabularies [{patch_mode}]"):
        feats, metas, summary = get_image_features(
            model=model,
            img_path=img_path,
            patch_mode=patch_mode,
            dataset_root=image_dir,
            filter_cfg=filter_cfg,
            return_summary=True,
        )

        page_manifest.append(summary)
        page_id = summary['page_id']

        if feats is None:
            continue
        if feats.shape[0] < VOCAB_SIZE:
            summary['excluded'] = True
            summary['exclusion_reason'] = f'features_lt_vocab_size(<{VOCAB_SIZE})'
            continue

        try:
            km = KMeans(n_clusters=VOCAB_SIZE, random_state=SEED, n_init=10).fit(feats)
        except Exception as e:
            summary['excluded'] = True
            summary['exclusion_reason'] = f'kmeans_error: {e}'
            continue

        clusters_data = [
            {
                'visual_word_index': i,
                'centroid': km.cluster_centers_[i],
                'patch_locations': [],
            }
            for i in range(VOCAB_SIZE)
        ]

        for patch_idx, label in enumerate(km.labels_):
            clusters_data[label]['patch_locations'].append(metas[patch_idx])

        page_data[page_id] = {
            'page_id': page_id,
            'vocabulary': km.cluster_centers_,
            'clusters_data': clusters_data,
            'img_path': img_path,
            'patch_mode': patch_mode,
            'num_raw_features': int(feats.shape[0]),
        }

    print(f"Built vocabularies for {len(page_data)} pages.")

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f'bob_page_data_{patch_mode}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(page_data, f)
    print(f"BoB data saved to {save_path}")

    if manifest_csv is None:
        manifest_csv = os.path.join(OUTPUT_DIR, f'page_filtering_log_{patch_mode}.csv')
    pd.DataFrame(page_manifest).to_csv(manifest_csv, index=False)
    print(f"Page filtering log saved to {manifest_csv}")

    return page_data


def make_folder_label_fn(page_data: dict) -> callable:
    """
    Build a label function that reads from the parent folder name.
    Falls back to filename parsing if the folder doesn't match cluster_NNN pattern.
    """
    from .distances import get_cluster_label
    lookup = {}
    for img_name, data in page_data.items():
        parent = os.path.basename(os.path.dirname(data['img_path']))
        lookup[img_name] = parent if re.match(r'cluster_\d+', parent) \
                           else get_cluster_label(img_name)
    return lambda name: lookup.get(name, get_cluster_label(name))


def validate_label_consistency(page_data: dict):
    """
    Cross-check filename labels vs folder labels.
    Prints a warning for any mismatch — catches accidental misplaced files.
    """
    from .distances import get_cluster_label
    mismatches = []
    for img_name, data in page_data.items():
        parent       = os.path.basename(os.path.dirname(data['img_path']))
        label_folder = parent if re.match(r'cluster_\d+', parent) else None
        label_file   = get_cluster_label(img_name)
        if label_folder and label_folder != label_file:
            mismatches.append((img_name, label_file, label_folder))

    if mismatches:
        print(f"\n⚠️  {len(mismatches)} LABEL MISMATCHES (filename ≠ folder):")
        for fname, lf, ld in mismatches[:10]:
            print(f"   {fname:50s}  file={lf}  folder={ld}")
    else:
        print("✅  All filename labels match folder labels.")
    return mismatches


def extract_all_features(
    model: SparseAutoencoder,
    image_dir: str,
    patch_mode: str = INFER_PATCH_MODE,
    cache_path: Optional[str] = None,
) -> dict:
    """
    Cache raw encoded features using stable page_id keys.
    """
    if cache_path is None:
        cache_path = os.path.join(OUTPUT_DIR, f'features_cache_{patch_mode}.pkl')

    if os.path.exists(cache_path):
        print(f"Loading feature cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    image_paths = _list_image_paths(image_dir)
    model.eval()
    cache = {}
    page_manifest = []
    filter_cfg = PatchFilterConfig()

    for img_path in tqdm(image_paths, desc=f"Caching features [{patch_mode}]"):
        feats, metas, summary = get_image_features(
            model=model,
            img_path=img_path,
            patch_mode=patch_mode,
            dataset_root=image_dir,
            filter_cfg=filter_cfg,
            return_summary=True,
        )
        page_manifest.append(summary)

        if feats is None:
            continue

        page_id = summary['page_id']
        cache[page_id] = (feats, metas)

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    pd.DataFrame(page_manifest).to_csv(
        os.path.join(OUTPUT_DIR, f'features_manifest_{patch_mode}.csv'),
        index=False
    )

    print(f"Cached {len(cache)} images → {cache_path}")
    return cache
