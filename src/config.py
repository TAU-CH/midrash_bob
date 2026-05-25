# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch

try:
    import ot          # pip install POT
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("POT not found. BoB-OT falls back to uniform Hungarian. Install: pip install POT")


# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TRAIN_PATCH_MODE = 'fitpad'   # main protocol: make train/infer symmetric
INFER_PATCH_MODE = 'fitpad'   # keep this as default for retrieval
PATCH_MODE_ABLATIONS = [
    ('context', 'fitpad'),
    ('fitpad',  'fitpad'),
]


# --- Paths ---
BASE_DIR        = '/content/drive/MyDrive/Genizah_joins'
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'joins_images_bin')    # binarized images for training
FEAT_IMAGE_DIR  = os.path.join(BASE_DIR, 'joins_images_bin')    # binarized images for feature extraction
COLOR_IMAGE_DIR = os.path.join(BASE_DIR, 'joins_images_color')  # color images for visualization
OUTPUT_DIR      = os.path.join(BASE_DIR, 'output_f1_200')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH     = os.path.join(OUTPUT_DIR, 'bob_model.pth')
BOB_DATA_FILE  = os.path.join(OUTPUT_DIR, 'bob_page_data.pkl')
BOW_DATA_FILE  = os.path.join(OUTPUT_DIR, 'bow_data.pkl')
RESULTS_CSV    = os.path.join(OUTPUT_DIR, 'retrieval_results.csv')
BCC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'bcc_clusters')


# --- Autoencoder Hyperparameters ---
PATCH_SIZE              = 64
ENCODED_DIM             = 128
BATCH_SIZE              = 256
EPOCHS                  = 30
LEARNING_RATE           = 1e-3
SPARSITY_WEIGHT         = 1e-5
PATCHES_PER_IMAGE_TRAIN = 300

# --- BoB Vocabulary Parameters ---
VOCAB_SIZE           = 20    # k in per-image KMeans
TARGET_FIT_SIZE      = 60    # max(w,h) after scaling inside patch canvas
MIN_AREA             = 300   # minimum CC area (pixels)
MAX_AREA             = 3000  # maximum CC area (pixels)
MIN_VALID_COMPONENTS = 200  # minimum valid CCs per image to include it

# --- Patch Quality Filters ---
MIN_COMPONENT_WHITE_RATIO = 0.05  # bounding-box crop: min fraction of white pixels
MIN_CANVAS_WHITE_RATIO    = 0.02  # after fit-and-pad canvas: min fraction of white

# --- Ablation search spaces ---
ABLATION_K_VALUES     = (8, 16, 20, 32, 64)
ABLATION_DIM_VALUES   = (64, 128, 256)
ABLATION_DISPLAY_COLS = ['Hit@1', 'mAP@1', 'Hit@5', 'mAP@5', 'MRR', 'MacroF1@1']

# --- Reranking ---
RERANK_TOP_M = 30

# --- BoW Baseline Parameters ---
BOW_VOCAB_SIZE = 100  # global codebook size K

# --- Evaluation ---
EVAL_KS = (1, 5, 10)

# --- BCC Graph Analysis ---
SIMILARITY_THRESHOLD = 0.48
MIN_BCC_SIZE         = 3
N_PAIRS_PER_CLUSTER  = 10
NUM_SAMPLES_TO_PLOT  = 5


BOW_CENTROIDS_DATA_FILE = os.path.join(OUTPUT_DIR, f'bow_centroids_{INFER_PATCH_MODE}.pkl')
BOW_RAWPATCH_DATA_FILE  = os.path.join(OUTPUT_DIR, f'bow_rawpatch_{INFER_PATCH_MODE}.pkl')

FLAT_POOL_FILE = os.path.join(OUTPUT_DIR, f'flat_pool_{INFER_PATCH_MODE}.pkl')
RUN_MAXPOOL = True
