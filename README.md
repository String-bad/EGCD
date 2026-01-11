# AD演化预测 - JiT扩散模型

基于JiT (Jacobian-informed Trajectory) 扩散模型预测阿尔茨海默病患者的未来MRI。

## 核心思想

```
JiT数学框架:
├── Flow:     z_t = t * x + (1-t) * ε,  ε ~ N(0,1)
├── 预测:     x̂ = model(z_t, t, condition)  
├── 损失:     v-loss = ||v_pred - v_target||²
└── 采样:     Heun ODE solver (从纯噪声到clean图像)

增强条件注入:
├── 历史图像直接concat到UNet输入 (4通道: z_t + 3个历史帧)
├── AdaGN调制 (与JiT的adaLN一致)
└── 临床信息融合
```

## 文件结构

```
ad_jit_clean/
├── train_jit.py      # 训练脚本 (包含模型定义)
├── inference.py      # 推理脚本
├── dataset.py        # 数据加载
├── preprocess_nifti.py  # NIfTI预处理工具
└── README.md
```

## 数据格式

### 目录结构
```
data_root/
├── images/
│   ├── subject_001/
│   │   ├── visit_0.png
│   │   ├── visit_1.png
│   │   ├── visit_2.png
│   │   └── visit_3.png
│   └── subject_002/
│       └── ...
```

### 临床数据CSV
```csv
subject_id,visit,age,sex,mmse,cdr,apoe,diagnosis,visit_month
subject_001,0,72.5,1,28,0.5,1,MCI,0
subject_001,1,73.5,1,27,0.5,1,MCI,12
...
```

## 使用方法

### 1. 数据预处理 (NIfTI → PNG)

```bash
python preprocess_nifti.py \
    --input_dir /path/to/nifti \
    --output_dir /path/to/png \
    --strategy hippocampus \
    --size 256
```

### 2. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_jit.py \
    --data_path /path/to/data \
    --clinical_csv /path/to/clinical.csv \
    --model_size B \
    --batch_size 4 \
    --epochs 300 \
    --lr 1e-4 \
    --eval_every 10 \
    --sample_steps 50 \
    --cfg_scale 2.0
```

### 3. 推理

```bash
python inference.py \
    --checkpoint ./output_jit/run_xxx/checkpoints/best.pt \
    --data_path /path/to/data \
    --output_dir ./results \
    --sample_steps 50 \
    --cfg_scale 2.0
```

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | 必需 | 数据目录 |
| `--clinical_csv` | None | 临床数据CSV |
| `--model_size` | B | 模型大小: S (~20M), B (~80M), L (~150M) |
| `--batch_size` | 4 | 批大小 |
| `--epochs` | 300 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--warmup_epochs` | 10 | 预热轮数 |
| `--eval_every` | 10 | 验证间隔 |
| `--save_every` | 50 | 保存间隔 |
| `--sample_steps` | 50 | 采样步数 |
| `--cfg_scale` | 2.0 | CFG强度 |
| `--patience` | 50 | 早停耐心值 |
| `--use_amp` | True | 混合精度训练 |
| `--no_clinical` | False | 禁用临床信息 |

### 模型配置

| 大小 | base_ch | 通道倍数 | 参数量 |
|------|---------|----------|--------|
| S | 48 | [1,2,4,4] | ~20M |
| B | 64 | [1,2,4,8] | ~80M |
| L | 96 | [1,2,4,8] | ~150M |

## 输出结构

```
output_jit/run_20241226_XXXXXX/
├── config.json           # 训练配置
├── checkpoints/
│   ├── best.pt          # 最佳模型 (PSNR最高)
│   └── epoch_0100.pt    # 定期保存
└── samples/
    └── epoch_0010/
        ├── sample_0.png # 对比图
        ├── sample_1.png
        └── ...
```

## 评估指标

- **PSNR** (Peak Signal-to-Noise Ratio): 越高越好, >25dB良好, >30dB优秀
- **SSIM** (Structural Similarity): 越接近1越好, >0.8良好

## 显存需求

| 配置 | 估计显存 |
|------|---------|
| S + batch=8 + AMP | ~10GB |
| B + batch=4 + AMP | ~12GB |
| B + batch=2 + AMP | ~8GB |
| L + batch=2 + AMP | ~16GB |

## 依赖

```
torch >= 1.10
numpy
pillow
pandas
tqdm
matplotlib
nibabel  # 仅预处理需要
```
