# Hierarchical EMG Classification Project

基于双通道 sEMG 的层级动作分类系统，区分哑铃弯举(1)、锤式弯举(2)、推肩(3)。

## 项目概览

```
文件 → 预处理 → 动作分割 → 40维特征提取 → 两层分类(SVM+RF) → 文件级决策
```

| 环节 | 方法 | 说明 |
|------|------|------|
| 预处理 | detrend → 50Hz陷波 → 20-500Hz带通 | 去基线+工频+噪声 |
| 分割 | 通道归一化能量 + 自适应阈值 + Top-6筛选 | P95归一化使两通道等权 |
| 特征 | 40维 (时域12+频域6)×2通道 + 跨通道4 | 丰富特征捕捉肌电模式 |
| 分类 | Stage1 SVM(RBF) + Stage2 RF + RF分数仲裁 | 层级决策+文件级投票 |
| 验证 | 70/30留出 + Stage2 5折交叉验证 | 双重评估 |

## Folder Layout

```
HierarchicalEMG_Project/
  run_project.m                 Main entry point
  config/
    project_config.m            All parameters
  src/
    io/                         File discovery, labels, MAT loading
    signal/                     EMG preprocessing
    segmentation/               Action segmentation
    features/                   Feature extraction (40-dim)
    model/                      Hierarchical SVM+RF + file-level aggregation
    evaluation/                 Metrics, confusion matrix
    visualization/              Diagnostic plots
  data/
    train/                      Training .mat files
    hidden_test/                Hidden test .mat files
  outputs/                      Generated results (timestamped folders)
  tests/                        Unit tests
```

## 数据命名规则

| 文件模式 | ID范围 | Label | 动作 |
|----------|--------|-------|------|
| `emg0xx.mat` | 1–99 | 1 | BicepsCurl |
| `emg1xx.mat` | 100–199 | 2 | HammerCurl |
| `emg2xx.mat` | 200–299 | 3 | ShoulderPress |

隐藏测试集无命名限制，所有 `.mat` 文件均处理。

## 特征设计

### 40维特征 (F1-F40)

```
F1-F12   二头肌时域: RMS, MAV, VAR, WL, ZC, SSC, WAMP, 偏度, 峰度, 峰值因子, 脉冲因子, IEMG
F13-F18  二头肌频域: MF, MDF, PF, 频谱熵, 带宽, 低频能量比
F19-F30  三头肌时域: (同上12维)
F31-F36  三头肌频域: (同上6维)
F37      RMS比值 (二头/三头)
F38      能量比值 (二头/三头)
F39      通道相关系数
F40      三头肌RMS
```

### 两阶段特征选择

| 阶段 | 特征索引 | 维度 | 特征 | 生理逻辑 |
|------|---------|------|------|----------|
| Stage 1 | [40,37,38] | 3 | 三头RMS, RMS比, 能量比 | 推肩=三头发力，比值低 |
| Stage 2 | 双通道+跨通道 | 24 | 时域+频域+跨通道 | 弯举/锤式的细微差异需全面特征 |

## 分类架构

```
输入 (40维特征)
      │
      ▼
┌──────────────────────┐
│  Stage 1 SVM (3维)   │  三头RMS + RMS比 + 能量比
│  推肩 vs 非推肩       │  ClassNames=[0,1], 1=推肩
└──────┬───────────────┘
       │
   ┌───┴───┐
  推肩(3)  非推肩(0)
           │
           ▼
   ┌──────────────────────┐
   │  Stage 2 RF (24维)    │  双通道时域+频域+跨通道
   │  弯举 vs 锤式弯举     │  50棵树, MinLeaf=5
   └──────────────────────┘
       │
       ▼
 ┌─────────────────┐
 │ 文件级决策        │  多数投票 + RF分数平票仲裁
 │ aggregate_file   │  弯举票差<2 → 用Stage2 RF平均分数决定
 └─────────────────┘
```

## 分割算法

**通道归一化自适应阈值**：

```
滑动RMS → 各通道÷P95 → 合并能量 sqrt(rms1n²+rms2n²)
→ 平滑 → 自适应阈值(median+0.20×动态范围)
→ 填小间隙(0.2s) → 滤短段(0.35s) → 前后填充
→ Top-6能量筛选 (每文件4-6次重复动作)
```

核心创新：**P95通道归一化**解决不同批次电极增益差异(可达20倍)，使两通道在能量计算中等权贡献。

## 文件级决策

`aggregate_file_prediction.m` 实现两阶段决策：

1. **多数投票** — 统计各片段预测标签，取多数
2. **RF分数仲裁** — 当弯举vs锤式弯举票差 < `cfg.fileVoteMinMargin`(默认2)时，用Stage2 RF的平均分类分数打破平局

输出详细信息包括每文件各标签票数、Stage2平均分数、决策规则。

## 过程总结：问题与解决

### 1. 推肩分类失败

**现象**: Ratio<1(三头>二头)却判为弯举。

**根因**: Stage1包含`XCorrLag_ms`(互相关时延)，训练集推肩低时延 → SVM学到"大时延=弯举"的错误规律，隐藏测试集推肩时延大被判错。

**解决**: Stage1仅保留生理相关的三头RMS+RMS比+能量比(3维)，排除时延。

### 2. 两通道幅值不对等

**现象**: 不同批次数据幅值差20倍，CH1可达CH2的100倍。合并能量被强通道主导，弱通道(三头肌)激活不可见 → 推肩窗口漏检。

**解决**: P95归一化——各通道RMS除以自身P95后再合并能量，两通道始终等权。

### 3. 分割碎片化

**现象**: `minGap=0.1s`太短 → 动作内RMS短暂回落被当边界 → 大量碎片；`sameActionGap=0.9s`过度补偿 → 相邻动作可能被合并。

**解决**: 调优参数(minGap→0.2s, smooth→0.12s, thr→0.20) + **Top-6能量筛选**（借鉴贺同学）：多于6段时按RMS能量排序只保留最强段。

### 4. 弯举/锤式弯举区分困难

**现象**: 两种弯举都是二头主导的拉动作，差异细微。最初9维特征不够 → 扩充至40维，包含偏度/峰度/频谱熵/带宽等形态描述特征。

**借鉴贺同学**: 每通道18维(时域12+频域6)×2+跨通道4维=40维，Stage2用24维(双通道+跨通道)，大幅提升特征信息量。

### 5. predict层class索引错误

**现象**: `curlIdx = (stage1Pred ~= 2)` — SVM class是[0,1]，`~=2`永远为真 → 推肩样本也被送入Stage2并覆盖为弯举。

**解决**: 改为`curlIdx = (stage1Pred == 0)`。

### 6. 特征过度工程化

**教训**: 早期尝试手工设计复杂特征(XCorr, nWL, 幅值无关特征)，效果反而不如回到标准EMG特征集(贺的40维)。**标准特征+充足维度 > 精心设计+维度不足**。

### 7. 文件级决策从简单到鲁棒

**演进**: `mode投票` → `平均特征向量` → `多数投票+SVM分数仲裁`

最终方案结合投票的鲁棒性和SVM分数的细粒度信息，在弯举票数接近时用分数决断。

## 创新点

1. **P95通道归一化能量** — 解决跨批/跨电极幅值差异，两通道在分割中等权
2. **Top-6能量筛选** — 用信号能量而非固定时间参数控制分割段数，对不同节奏鲁棒
3. **RF分数平票仲裁** — 文件级决策不只用离散投票，结合连续分数信息
4. **Stage2五折交叉验证** — 评估模型泛化能力，附带逐文件错误诊断
5. **分层特征设计** — Stage1用生理驱动的精简特征(3维)，Stage2用全面的双通道特征(24维)

## 使用

```matlab
% 配置 project_config.m
cfg.forceRetrain = true;   % 改参数后强制重训
cfg.showHiddenPlots = true; % 查看分割诊断图

% 运行
run_project
```

## 依赖

- MATLAB R2019b+
- Statistics and Machine Learning Toolbox (`fitcsvm`, `TreeBagger`, `cvpartition`, `prctile`)
- Signal Processing Toolbox (`fft`, `movmean`, `lowpass`, `highpass`)
