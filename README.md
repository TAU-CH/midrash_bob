<div align="center">

<h1>🎒 Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval</h1>

[Sharva Gogawale](),
[Gal Grudka](),
[Daria Vasyutinsky-Shapira](),
[Omer Ventura](),
[Berat Kurar-Barakat](),
[Nachum Dershowitz]()

**School of Computer Science and AI, Tel Aviv University, Ramat Aviv, Israel**

</div>

## 📋 Abstract

A join is a set of manuscript fragments identified as originally emanating from the same manuscript. We study manuscript join retrieval: Given a query image of a fragment, retrieve other fragments originating from the same physical manuscript. We propose **Bag of Bags (BoB)**, an image-level representation that replaces the global-level visual codebook of classical Bag of Words (BoW) with a fragment-specific vocabulary of local visual words. Our pipeline trains a sparse convolutional autoencoder on binarized fragment patches, encodes connected components from each page, clusters the resulting embeddings with per-image k-means, and compares images using set-to-set distances between their local vocabularies. Evaluated on fragments from the Cairo Genizah, the best BoB variant (viz. Chamfer) achieves Hit@1 of 0.78 and MRR of 0.84, compared to 0.74 and 0.80, respectively, for the strongest BoW baseline (BoW-RawPatches-χ²), a 6.1% relative improvement in top-1 accuracy. We furthermore study a mass-weighted BoB-OT variant that incorporates cluster population into prototype matching, and present a formal approximation guarantee bounding its deviation from full component-level optimal transport. A two-stage pipeline using a BoW shortlist followed by BoB-OT reranking provides a practical compromise between retrieval strength and computational cost, supporting applicability to larger manuscript collections.

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (falls back to CPU automatically)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/midrash_bob.git
cd midrash_bob

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `POT` (Python Optimal Transport) is required only for the `BoB-OT` variant.  
> If it is not installed, the pipeline automatically falls back to the Hungarian-L2 distance.

## 🚀 Getting Started

### 1. Configure paths

Open `src/config.py` and update `BASE_DIR` to point to your local data directory:

```python
BASE_DIR = '/path/to/your/Genizah_joins'   # ← change this
```

All output sub-directories (`OUTPUT_DIR`, `COLOR_IMAGE_DIR`, etc.) are derived from `BASE_DIR` and will be created automatically.

### 2. Prepare data

Place your binarized manuscript fragment images (`.png` / `.jpg`) under the paths referenced by `TRAIN_IMAGE_DIR` and `FEAT_IMAGE_DIR` in `config.py`.  
Images should be organised in sub-folders named after their join cluster (e.g. `cluster_01/fragment_a.png`).

### 3. Run the full pipeline

```bash
python -m src.main
```

This will:
1. Train (or load a cached) sparse convolutional autoencoder
2. Build per-image BoB vocabularies with KMeans
3. Compute pairwise BoB distance matrices (Hungarian-L2, Chamfer, optionally BoB-OT)
4. Build BoW-Centroids, BoW-RawPatches, and flat-pool baselines
5. Print a full retrieval evaluation table (Hit@k, mAP@k, MRR, MacroF1@1) and save it to `results/results.csv`
6. Generate distance distribution plots and runtime profiling
7. Run ablation studies (vocabulary size, encoder dimension, sparsity weight)
8. Run BCC graph analysis and save visualisations for discovered joins

## 🗂️ Module Overview

| Module | Description |
|---|---|
| `src/config.py` | All hyperparameters and file paths |
| `src/model.py` | `SparseAutoencoder` (convolutional encoder–decoder) |
| `src/dataset.py` | Patch extraction, `TextPatchDataset`, autoencoder training |
| `src/features.py` | Patch encoding, per-image BoB vocabulary building |
| `src/baselines.py` | BoW-Centroids, BoW-RawPatches, flat-pool baselines |
| `src/distances.py` | Hungarian-L2, Chamfer, Hungarian-Cosine, BoB-OT distance matrices |
| `src/evaluation.py` | Hit@k, mAP@k, MRR, MacroF1@1; full evaluation table |
| `src/utils.py` | Shared image loading/caching and patch reconstruction |
| `src/visualization.py` | Matched-pair visualization, distance separation stats |
| `src/bcc.py` | BCC graph analysis, top-k retrieval grid plots, matched-pair figures |
| `src/ablations.py` | Vocabulary size, encoder dim, sparsity ablations; distance distribution plots, runtime profiling |
| `src/main.py` | End-to-end pipeline entry point |

## 📊 Output Files

After a successful run you will find the following under `OUTPUT_DIR` (default: `BASE_DIR/output`):

| File | Contents |
|---|---|
| `bob_model_fitpad.pth` | Trained autoencoder weights |
| `bob_page_data_fitpad.pkl` | Per-page BoB vocabulary dictionaries |
| `bob_distance_matrix_hungarian_l2.npz` | BoB distance matrix |
| `bow_centroids_data.pkl` | BoW-Centroids representation |
| `bow_rawpatch_data.pkl` | BoW-RawPatches representation |
| `results/results.csv` | Full retrieval evaluation table |
| `fig_reconstructions_fitpad.png` | Autoencoder reconstruction grid |

## 🗃️ Data

Arrange your dataset so that each **join cluster** (i.e. class) is a separate sub-folder, and each folder contains fragment images (`.jpg` / `.png`).

```
data/
├── cluster_1/
│   ├── fragment_001.jpg
│   ├── fragment_002.jpg
│   └── ...
├── cluster_2/
│   ├── fragment_010.jpg
│   └── fragment_011.jpg
├── cluster_3/
│   └── fragment_020.png
└── ...
```

- **Folder name** → treated as the ground-truth join label for evaluation.  
- **Images inside** → individual manuscript fragment scans belonging to that join.

Then point `BASE_DIR` in `src/config.py` to the parent directory that contains these cluster folders (see [Configure paths](#1-configure-paths) above).

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{Gogawale_2026_CVPR,
    author    = {Gogawale, Sharva and Grudka, Gal and Vasyutinsky-Shapira, Daria and Ventura, Omer and Kurar-Barakat, Berat and Dershowitz, Nachum},
    title     = {Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {1-9}
}
```
