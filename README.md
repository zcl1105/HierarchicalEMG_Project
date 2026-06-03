# Hierarchical EMG Classification Project

This is the engineering version of the hierarchical sEMG action classifier.

## Folder Layout

```text
HierarchicalEMG_Project/
  run_project.m                 Main entry point
  config/
    project_config.m            Project paths and parameters
  src/
    io/                         File discovery, labels, MAT loading
    signal/                     EMG preprocessing and notch filter
    segmentation/               Action segmentation
    features/                   Feature extraction
    model/                      Hierarchical SVM training and prediction
    evaluation/                 Test-set reporting
    visualization/              Diagnostic plots
  data/
    train/                      Training .mat files
    hidden_test/                Put hidden test .mat files here
  outputs/                      Generated Excel tables and model files
```

The current training files are in `data/train/legacy_data/` (subdirectories 数据二/三/四).
Files are discovered recursively — any `.mat` file under `data/train/` with a matching name
pattern is automatically included.

## Data Naming Rule

Training labels are inferred from the numeric part of the file name via regex `^emg(\d+)$`:

| Numeric ID range | Label | Action |
|-----------------|-------|--------|
| 1–99 (`emg0xx`) | 1 | BicepsCurl |
| 100–199 (`emg1xx`) | 2 | HammerCurl |
| 200–299 (`emg2xx`) | 3 | ShoulderPress |

Files not matching the pattern (e.g. `1.mat`, `36.mat`) are ignored during training
but are still processed in `data/hidden_test/` for prediction.

## Outputs

After running `run_project.m`, a timestamped folder is created under `outputs/`:

```
outputs/
  LATEST.txt                       Points to most recent model version
  <yyyymmdd_HHMMSS>/
    dataset.xlsx                   Full feature dataset
    model.mat                      Trained hierarchical SVM models
    segment_template.xlsx          Segmentation windows (for manual override)
    hidden_detail.xlsx             Per-segment hidden test predictions
    Pred_Labels.csv                Hidden test submission file
```

A cached model is loaded automatically on subsequent runs. Set `cfg.forceRetrain = true`
in [config/project_config.m](config/project_config.m) to force a full retrain.

## Dataset Split

- Training set: 70% of samples from `data/train/`, used to fit both SVM models.
- Test set: 30% hold-out from `data/train/`, used for accuracy evaluation.
- Hidden test set: loaded from `data/hidden_test/`, used only for final prediction.

Split uses `cvpartition` with `HoldOut` and `cfg.testRatio` (default 0.30).
Random seed is fixed by `cfg.randomSeed` for reproducibility.

## Segmentation

### Algorithm

The segmentation pipeline uses **channel-normalized adaptive thresholding**:

```
原始EMG → 滑动RMS → 通道归一化(rms/P95) → 合并能量 → 平滑
→ 自适应阈值 → 填小间隙 → 滤短片段 → 前后填充 → 合并重叠 → 合并双峰
```

**Key insight**: Each channel's RMS is divided by its own 95th percentile before
combination (`energy = sqrt((rms1/P95₁)² + (rms2/P95₂)²)`). This ensures both
channels contribute equally to the energy curve regardless of absolute amplitude
differences between recordings or electrode placements.

### Manual segmentation

After one run, the project exports `outputs/<timestamp>/segment_template.xlsx`.
Edit the `StartTime_s` and `EndTime_s` columns, save as
`data/manual_segments.xlsx`, and the next run will use those windows instead of
auto-segmentation for matching files.

## Hierarchical Classification

```
输入 (9维特征)
      │
      ▼
┌──────────────────────┐
│  Stage 1 SVM (4维)   │  RMS1, RMS2, Ratio, XCorrCoef
│  推肩 vs 非推肩       │
└──────┬───────────────┘
       │
   ┌───┴───┐
  推肩(3)  非推肩
           │
           ▼
   ┌──────────────────────┐
   │  Stage 2 SVM (9维)   │  全部特征
   │  弯举 vs 锤式弯举     │
   └──────────────────────┘
```

- Stage 1 trained on **all** samples; Stage 2 trained on **curl-only** samples.
- 9-dim feature: `RMS1, RMS2, Ratio, XCorrCoef, XCorrLag_ms, MF1, MF2, nWL1, nWL2`

## Plotting

Two independent switches in [project_config.m](config/project_config.m):

| Switch | Controls |
|--------|----------|
| `cfg.showTrainPlots` | Segmentation diagnostic plots for training files |
| `cfg.showHiddenPlots` | Segmentation diagnostic plots for hidden test files (with prediction labels) |

Hidden test plots show predicted class names on each action window.

## Troubleshooting & Changelog

### 两通道幅值不对等导致弱通道被淹没

**现象**: 不同批次的 EMG 数据幅值差异高达 20 倍，同一文件中 CH1 可达 CH2 的 10–100 倍。合并能量 `sqrt(rms1² + rms2²)` 完全由强通道主导，弱通道（通常为肱三头肌）的激活在能量曲线上不可见，导致推肩动作的窗口检测失败。

**解决**: `segment_actions.m` 中改为通道归一化能量——各通道 RMS 先除以自身 P95，再合并。两通道始终等权贡献，与绝对幅值无关。

### 分割参数不当导致碎片化

**解决**: 调整参数（详见 [project_config.m](config/project_config.m) 注释）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `energySmooth_s` | 0.12 | 平滑窗宽，减少碎片 |
| `thresholdRatio` | 0.20 | 自适应阈值比例，检出弱激活 |
| `minGap_s` | 0.20 | 填小间隙，避免碎片化 |
| `minDuration_s` | 0.35 | 最短动作时长，过滤噪声 |
| `sameActionGap_s` | 0.75 | 同动作双峰合并间距 |

### 整文件模式

设置 `cfg.useSegmentation = false` 可跳过动作窗口切割，将每个 .mat 文件整体作为一个样本提取特征。适合快速验证分类器效果。

### 缓存模型导致修改不生效

**现象**: 改了参数或特征但预测结果不变。

**解决**: 设置 `cfg.forceRetrain = true` 强制重新训练。训练完成后建议改回 `false`。

## Requirements

- MATLAB R2019b or later
- Statistics and Machine Learning Toolbox (`fitcsvm`, `cvpartition`, `prctile`)
- Signal Processing Toolbox (`xcorr`, `fft`, `movmean`, `lowpass`, `highpass`)

## Run

Open MATLAB in this folder and run:

```matlab
run_project
```

The script automatically adds `src/` and `config/` to the MATLAB path.
