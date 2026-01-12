# ReFlow: Residual Flow Matching for Longitudinal Brain MRI Generation

Official PyTorch implementation of **"Learning What Changes: Residual Flow Matching for Longitudinal Brain MRI Generation"**.

## 🔑 Key Features

- **Residual Learning**: Only learns the subtle structural changes (1-3% of the image), not the entire brain anatomy
- **Flow Matching**: Efficient generation with straight probability paths (50 steps vs. 1000 for diffusion)
- **Multi-scale Temporal Conditioning**: Captures frame-level anatomy, inter-scan progression trends, and clinical biomarkers
- **3D Native**: Full 3D volumetric processing for brain MRI
- **Interpretable Outputs**: Predicted residuals directly visualize disease-induced changes

## 📁 Project Structure

```
ReFlow/
├── train_jit_3d_residual.py    # Training script
├── inference_3d_residual.py    # Inference script
├── dataset_3d.py               # Dataset classes
├── preprocess_3d.py            # NIfTI preprocessing
├── requirements.txt            # Dependencies
├── configs/                    # Configuration files
│   └── default.yaml
├── assets/                     # Images for README
└── checkpoints/                # Saved models
```

## 🛠️ Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/String-bad/EGCD.git
cd EGCD

# Create conda environment
conda create -n reflow python=3.10
conda activate reflow

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
numpy>=1.21.0
nibabel>=5.0.0
scipy>=1.9.0
pandas>=1.5.0
tqdm>=4.64.0
scikit-image>=0.19.0
matplotlib>=3.6.0
```

## 📊 Data Preparation

### Expected Data Structure

```
data_root/
├── sub-001/
│   ├── ses-20060418082030/
│   │   └── anat/
│   │       └── sub-001_ses-xxx_T1w.nii.gz
│   ├── ses-20061102081644/
│   │   └── anat/
│   │       └── sub-001_ses-xxx_T1w.nii.gz
│   └── ...
├── sub-002/
│   └── ...
└── clinical_data.csv
```

Or ADNI format:
```
data_root/
├── 002_S_0619/
│   ├── bl.nii.gz      # baseline
│   ├── m12.nii.gz     # month 12
│   ├── m24.nii.gz     # month 24
│   └── ...
└── ...
```

### Preprocessing

```bash
python preprocess_3d.py \
    --input_dir /path/to/raw/nifti \
    --output_dir /path/to/processed \
    --volume_size 96 112 96 \
    --skull_strip \
    --bias_correction \
    --n_jobs 8
```

### Clinical Data Format (Optional)

CSV file with columns:
| Column | Description |
|--------|-------------|
| `subject_id` or `RID` | Subject identifier |
| `VISCODE2` | Visit code (bl, m12, m24, ...) |
| `MMSE` | Mini-Mental State Examination score |
| `CDRSB` | CDR Sum of Boxes |
| `AGE` | Age at visit |
| `PTGENDER` | Sex (Male/Female) |
| `APOE4` | APOE ε4 allele count (0/1/2) |

## 🚀 Training

### Basic Training

```bash
python train_jit_3d_residual.py \
    --data_path /path/to/processed/data \
    --clinical_csv /path/to/clinical.csv \
    --output_dir ./output \
    --model_size S \
    --num_history 3 \
    --batch_size 1 \
    --grad_accum 4 \
    --epochs 300 \
    --lr 1e-4
```

### Model Sizes

| Size | Base Channels | Parameters | GPU Memory |
|------|---------------|------------|------------|
| S    | 32            | ~15M       | ~8GB       |
| M    | 48            | ~40M       | ~16GB      |
| L    | 64            | ~80M       | ~24GB      |

### Training Options

```bash
# Full options
python train_jit_3d_residual.py \
    --data_path /path/to/data \
    --clinical_csv /path/to/clinical.csv \
    --output_dir ./output \
    --model_size S \
    --num_history 3 \
    --volume_size 96 112 96 \
    --max_time_delta 120 \
    --random_target \
    --batch_size 1 \
    --grad_accum 4 \
    --epochs 300 \
    --lr 1e-4 \
    --warmup 1000 \
    --use_clinical \
    --use_checkpoint \
    --use_amp \
    --sample_steps 50 \
    --cfg_scale 2.0 \
    --eval_every 10 \
    --save_every 50 \
    --patience 50
```

### Ablation Studies

```bash
# Without time delta conditioning
python train_jit_3d_residual.py ... --ablate_time_delta

# Without difference encoding
python train_jit_3d_residual.py ... --ablate_diff

# Without clinical features
python train_jit_3d_residual.py ... --no_clinical
```

## 🔮 Inference

### Basic Inference

```bash
python inference_3d_residual.py \
    --checkpoint ./output/run_xxx/checkpoints/best.pt \
    --data_path /path/to/data \
    --output_dir ./results
```

### Predict at Specific Time Point

```bash
# Predict 12 months ahead
python inference_3d_residual.py \
    --checkpoint ./output/run_xxx/checkpoints/best.pt \
    --data_path /path/to/data \
    --time_delta 12 \
    --output_dir ./results_12m
```

### Inference Options

```bash
python inference_3d_residual.py \
    --checkpoint /path/to/best.pt \
    --data_path /path/to/data \
    --clinical_csv /path/to/clinical.csv \
    --output_dir ./results \
    --time_delta 12 \           # Fixed time delta (months)
    --num_samples -1 \          # -1 for all samples
    --sample_steps 50 \         # Euler steps
    --cfg_scale 2.0 \           # Classifier-free guidance
    --fast                      # Use fast Euler (vs Heun)
```

### Output Structure

```
results/
├── nifti/                      # Generated NIfTI files
│   ├── subj_001_m24_generated.nii.gz
│   └── subj_001_m24_gt.nii.gz
├── residuals/                  # Predicted residuals (for visualization)
│   └── subj_001_m24_residual.nii.gz
├── comparisons/                # Visual comparisons
│   └── subj_001_m24_comparison.png
├── metrics.csv                 # Per-sample metrics
├── summary.json                # Aggregated statistics
└── results_table.tex           # LaTeX table
```
