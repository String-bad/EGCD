# Longitudinal Medical Image Generation Framework

A PyTorch framework for predicting future medical images from historical longitudinal data using JiT-style flow matching. Designed with strict scientific constraints to ensure reproducibility and prevent information leakage.

## Overview

Given historical brain MRI images at multiple timepoints plus clinical features, this framework generates the predicted image at a future target timepoint.

```
Input: history_images (T frames) + clinical_features + delta_days
Output: predicted target_image at future timepoint
```

### Training Objective (JiT v-loss)

The training uses flow matching with v-loss:

```
z = t * x + (1 - t) * e          # Interpolation between noise and clean
x_pred = net(z, t, cond)         # Network predicts clean image
v = (x - z) / clamp(1-t, t_eps)  # Target velocity
v_pred = (x_pred - z) / clamp(1-t, t_eps)
loss = MSE(v, v_pred)
```

Reference: [JiT (Just image Transformers)](https://arxiv.org/abs/2511.13720)

## Key Design Principles

### 1. No Information Leakage
- **Clinical data alignment**: Only clinical records with `EXAMDATE <= img_date` are used
- **Target clinical exclusion**: Target timepoint clinical data is NOT available by default
- **Baseline-only diagnosis**: Uses `DX_bl` (baseline diagnosis), not `DX` (current diagnosis)

### 2. Reproducibility
- Fixed random seeds (Python, NumPy, PyTorch, CUDA)
- Pre-built index files (`index.jsonl`) - no runtime directory scanning
- Pre-computed subject splits (`splits.json`)
- Saved normalization statistics (`clinical_stats.json`)

### 3. No Patch-Based Training
- **Full image input only** (no random crops, no patch extraction)
- Optional center crop + resize to fixed `img_size`
- Consistent spatial dimensions across all samples

### 4. Explicit Missing Value Handling
- Binary mask for every feature (`1=present`, `0=missing`)
- Robust z-score normalization (median/IQR-based)
- Categorical fields use explicit `<UNK>` token (index 0)

---

## Quick Start

### 1. Build Index (if not already done)

```bash
# Build index.jsonl from raw data directory
python build_index.py \
    --data_root /path/to/DATA_ROOT \
    --clinical_csv /path/to/clinical.csv \
    --output_dir data/ \
    --history_len 2 \
    --min_timepoints 4 \
    --seed 42
```

### 2. Train Model

```bash
# Basic training
python train.py \
    --index_path data/index.jsonl \
    --splits_path data/splits.json \
    --output_dir outputs/exp1 \
    --batch_size 8 \
    --total_steps 100000 \
    --lr 1e-4

# Full training with all options
python train.py \
    --index_path data/index.jsonl \
    --splits_path data/splits.json \
    --stats_path data/clinical_stats.json \
    --output_dir outputs/exp1 \
    --img_size 256 \
    --history_len 2 \
    --batch_size 8 \
    --lr 1e-4 \
    --total_steps 100000 \
    --warmup_steps 1000 \
    --loss_type v \
    --P_mean -0.8 \
    --P_std 0.8 \
    --noise_scale 1.0 \
    --use_ema \
    --ema_decay 0.9999 \
    --seed 42
```

### 3. Run Inference & Evaluation

```bash
# Evaluate on test set
python inference.py \
    --checkpoint outputs/exp1/best.ckpt \
    --split test \
    --output_dir results/test \
    --num_steps 50 \
    --save_images

# Quick evaluation (fewer steps)
python inference.py \
    --checkpoint outputs/exp1/best.ckpt \
    --split val \
    --output_dir results/val \
    --num_steps 20 \
    --max_samples 100
```

---

## Ablation Studies

The framework supports three key ablations:

### 1. Without Clinical Features

```bash
python train.py \
    --index_path data/index.jsonl \
    --splits_path data/splits.json \
    --output_dir outputs/no_clinical \
    --no_clinical
```

### 2. Without Delta Days

```bash
python train.py \
    --index_path data/index.jsonl \
    --splits_path data/splits.json \
    --output_dir outputs/no_delta \
    --no_delta_days
```

### 3. Different History Length

```bash
# History length = 3 (requires index built with history_len=3)
python train.py \
    --index_path data/index_h3.jsonl \
    --splits_path data/splits.json \
    --output_dir outputs/hist3 \
    --history_len 3
```

---

## Training Configuration

### Flow Matching Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss_type` | `v` | Loss type: `v` (velocity) or `x0` (direct) |
| `--t_distribution` | `logit_normal` | Time sampling: `logit_normal` or `uniform` |
| `--P_mean` | `-0.8` | Mean for logit-normal distribution |
| `--P_std` | `0.8` | Std for logit-normal distribution |
| `--noise_scale` | `1.0` | Noise magnitude |
| `--t_eps` | `1e-3` | Stability clamp for 1-t division |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_channels` | `64` | Base channel count |
| `--channel_mult` | `1 2 4 8` | Channel multipliers per level |
| `--num_res_blocks` | `2` | ResBlocks per level |
| `--attention_levels` | `2 3` | Levels with self-attention |
| `--clinical_dim` | `256` | Clinical encoder output dim |
| `--cond_dim` | `512` | Fusion output dim |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--total_steps` | `100000` | Total training steps |
| `--warmup_steps` | `1000` | LR warmup steps |
| `--use_ema` | `True` | Use EMA model averaging |
| `--ema_decay` | `0.9999` | EMA decay rate |

---

## Inference Configuration

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_steps` | `50` | Number of ODE integration steps |
| `--sampler` | `euler` | Sampler type: `euler` or `heun` |

**Note**: More steps generally give better quality but take longer. Recommended:
- Quick evaluation: 20-30 steps
- Final results: 50-100 steps

The sampler does NOT use classifier-free guidance (CFG scale = 1.0) since this is conditional generation without unconditional training.

---

## Output Files

### Training Outputs (`outputs/exp1/`)

```
outputs/exp1/
├── config.json        # Full training configuration
├── train_stats.csv    # Training metrics over time
├── best.ckpt          # Best validation checkpoint
└── last.ckpt          # Last checkpoint
```

### Inference Outputs (`results/test/`)

```
results/test/
├── metrics.csv        # Per-sample PSNR/SSIM
├── summary.json       # Aggregated metrics
└── images/
    ├── sub-001_2020-01-01_pred.png
    ├── sub-001_2020-01-01_gt.png
    ├── sub-001_2020-01-01_residual.png
    ├── sub-001_2020-01-01_hist0.png
    └── sub-001_2020-01-01_hist1.png
```

### metrics.csv Format

```csv
subject_id,target_date,delta_days,psnr,ssim
sub-002_S_0295,2007-05-25,204,28.45,0.8923
sub-002_S_0295,2008-07-23,425,27.12,0.8756
...
```

---

## Dataset Format

### Directory Structure (Original Data)
```
DATA_ROOT/
├── sub-XXX/                        # Subject folder
│   ├── ses-YYYYMMDDHHMMSS/         # Session folder (date in name)
│   │   └── *.png                   # Single-channel brain image
│   └── ...
└── ...
```

### Index Files

See [Data Format Documentation](#data-format) below for detailed schemas.

---

## Dataset Output Format

`LongitudinalIndexedDataset.__getitem__(idx)` returns:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `history_images` | `float32` | `[T, 1, H, W]` | T historical frames (grayscale) |
| `target_image` | `float32` | `[1, H, W]` | Target frame to predict |
| `delta_days` | `float32` | scalar | Normalized time delta (days / 365) |
| `delta_days_raw` | `int64` | scalar | Raw delta in days |
| `time_features` | `float32` | `[T+1]` | `day_from_first` for each timepoint |
| `clinical_cont_seq` | `float32` | `[T, Fc]` | Continuous features (z-normalized) |
| `clinical_cat_seq` | `int64` | `[T, Fd]` | Categorical indices |
| `clinical_mask_seq` | `float32` | `[T, F]` | Missing mask (1=present, 0=missing) |
| `meta` | `dict` | - | Metadata (subject_id, dates, paths) |

**Default dimensions:**
- `T = 2` (history_len)
- `H = W = 256` (img_size)
- `Fc = 19` (continuous features)
- `Fd = 3` (categorical features: PTGENDER, APOE4, DX_bl)
- `F = Fc + Fd = 22`

---

## Architecture

### Model Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
├─────────────────────────────────────────────────────────────┤
│  history_images [B,T,1,H,W] ──┐                              │
│                               │                              │
│  clinical_cont_seq [B,T,Fc] ──┼──► ClinicalEncoder ──┐       │
│  clinical_cat_seq  [B,T,Fd] ──┤                      │       │
│  clinical_mask_seq [B,T,F]  ──┘                      │       │
│                                                      ▼       │
│  delta_days [B] ──────────────────────────► ConditionFusion │
│                                                      │       │
│                                                      ▼       │
│                                              cond_spatial    │
│                                              cond_vector     │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Conditional U-Net                         │
├─────────────────────────────────────────────────────────────┤
│  z [B,1,H,W] + cond_spatial [B,T,H,W] ──► Encoder            │
│                                              │               │
│  t [B] ────► Time Embedding ──┐              │               │
│  cond_vector [B,D] ───────────┼──► AdaGN ────┼──► Middle     │
│                               │              │               │
│                               └──────────────┼──► Decoder    │
│                                              │               │
│                                              ▼               │
│                                        x_pred [B,1,H,W]      │
└─────────────────────────────────────────────────────────────┘
```

### Clinical Features (Default)

**Continuous** (z-score normalized):
```
AGE, MMSE, CDRSB, ADAS11, ADAS13, FAQ,
Ventricles, Hippocampus, WholeBrain, Entorhinal,
Fusiform, MidTemp, ICV, PTEDUCAT,
RAVLT_immediate, RAVLT_learning, RAVLT_forgetting,
EcogPtTotal, EcogSPTotal
```

**Categorical** (embedded):
```
PTGENDER: {<UNK>: 0, Male: 1, Female: 2}
APOE4: {<UNK>: 0, 0: 1, 1: 2, 2: 3}
DX_bl: {<UNK>: 0, CN: 1, MCI: 2, AD: 3, SMC: 4, EMCI: 5, LMCI: 6, Dementia: 7}
```

---

## Data Format

### `index.jsonl`
Each line is a JSON object representing one training sample:
```json
{
  "subject_id": "sub-002_S_0295",
  "dataset_name": "AD_CN_2D",
  "history": [
    {
      "png_path": "/path/to/image_t0.png",
      "img_date": "2006-04-18",
      "day_from_first": 0,
      "session_id": "ses-20060418082030"
    },
    {
      "png_path": "/path/to/image_t1.png",
      "img_date": "2006-11-02",
      "day_from_first": 198,
      "session_id": "ses-20061102081644"
    }
  ],
  "target": {
    "png_path": "/path/to/target_image.png",
    "img_date": "2007-05-25",
    "day_from_first": 402,
    "session_id": "ses-20070525072037"
  },
  "delta_days": 204,
  "clinical_rows": [...]
}
```

### `splits.json`
```json
{
  "seed": 42,
  "split_ratio": [0.8, 0.1, 0.1],
  "train_subjects": ["sub-001", "sub-002", ...],
  "val_subjects": ["sub-050", ...],
  "test_subjects": ["sub-060", ...]
}
```

---

## Running Tests

```bash
# With real data
python tests/test_dataset_shapes.py \
    --index data/index.jsonl \
    --splits data/splits.json \
    --img-size 256 \
    --history-len 2

# With mock data (no real images needed)
python tests/test_dataset_shapes.py --use-mock
```

---

## Directory Structure

```
longitudinal_generation/
├── dataset_indexed.py       # Dataset module
├── diffusion_or_flow.py     # Flow matching loss & sampling
├── train.py                 # Training script
├── inference.py             # Inference & evaluation
├── metrics.py               # PSNR/SSIM metrics
├── models/
│   ├── __init__.py
│   ├── clinical_encoder.py  # Clinical feature encoder
│   ├── condition_fusion.py  # Condition fusion module
│   └── unet_cond.py         # Conditional U-Net backbone
├── tests/
│   ├── __init__.py
│   └── test_dataset_shapes.py
├── configs/
│   └── default.yaml         # Example config
└── README.md
```

---

## Requirements

```
torch>=2.0
torchvision
numpy
pillow
tqdm
```

---

## Citation

If you use this framework, please cite:

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

---

## License

Research use only. Please cite appropriately when using this framework.
