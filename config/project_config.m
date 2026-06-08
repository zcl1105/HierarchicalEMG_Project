function cfg = project_config(projectRoot)
% 项目全局配置文件
% 所有参数集中在这里，修改后设 cfg.forceRetrain=true 再运行

%% ---- 基础设置 ----
cfg.projectRoot = projectRoot;    % 项目根目录 (由 run_project 自动传入)
cfg.fs = 2000;                    % 采样率 (Hz), 硬件固定 2000
cfg.randomSeed = 42;              % 随机种子, 固定可复现结果, 改值可换一种随机划分

%% ---- 绘图开关 ----
cfg.showTrainPlots  = false;      % 是否显示训练集分割诊断图 (调试用, 批量跑时关掉)
cfg.showHiddenPlots = false;      % 是否显示隐藏测试集分割诊断图 (含预测标签)

%% ---- 强制重训 ----
cfg.forceRetrain = true;          % true=跳过缓存, 重新训练; false=加载 outputs/ 下最新模型

%% ---- 版本管理 ----
cfg.runTimestamp = datestr(now, 'yyyymmdd_HHMMSS');  % 运行时间戳, 用于输出目录命名
cfg.outputDir = fullfile(projectRoot, 'outputs', cfg.runTimestamp);  % 本次输出目录
cfg.latestLink = fullfile(projectRoot, 'outputs', 'LATEST.txt');     % 指向最新模型的指针文件

%% ---- 标签 & 特征索引 ----
cfg.classNames = {'BicepsCurl','HammerCurl','ShoulderPress'};  % 1=弯举 2=锤式弯举 3=推肩

% 40维特征: ch1时域(1:12) ch1频域(13:18) ch2时域(19:30) ch2频域(31:36) 跨通道(37:40)
%   时域12维: RMS, MAV, VAR, WL, ZC, SSC, WAMP, 偏度, 峰度, 峰值因子, 脉冲因子, IEMG
%   频域6维:  MF, MDF, PF, 频谱熵, 带宽, 低频能量比
%   跨通道4维: RMS比值, 能量比值, 通道相关系数, 三头肌RMS

cfg.stage1FeatureIdx = [40, 37, 38];
    % Stage 1: 推肩检测 → 三头RMS(40) + RMS比值(37) + 能量比值(38), 共3维
    % 推肩是"推"动作, 三头肌主导; 弯举是"拉"动作, 二头肌主导 → 三头激活+比值天然区分

cfg.stage2FeatureIdx = [1, 2, 4, 5, 6, 7, 13, 14, 16, 18, ...
                        19, 20, 22, 23, 24, 25, 31, 32, 34, 36, ...
                        37, 38, 39, 40];
    % Stage 2: 弯举细分 → 双通道时域+频域+跨通道, 共24维
    % 弯举/锤式弯举差异细微, 需要全面特征让RF自动学习

cfg.fileVoteMinMargin = 2;
    % 文件级决策: 弯举票差小于此值时, 不用多数投票, 改用Stage2 RF平均分数仲裁
    % 设为 1: 票差≥1就信投票 (几乎不仲裁)
    % 设为 99: 永远仲裁 (完全信RF分数)

%% ---- 路径配置 ----
cfg.trainDirs = {fullfile(projectRoot, 'data', 'train')};            % 训练数据目录 (递归搜索.mat)
cfg.hiddenTestDirs = {fullfile(projectRoot, 'data', 'hidden_test')}; % 隐藏测试集目录
cfg.manualSegmentPath = fullfile(projectRoot, 'data', 'manual_segments.xlsx');  % 手动分割文件(可选)

% 输出文件路径
cfg.datasetPath           = fullfile(cfg.outputDir, 'dataset.xlsx');           % 特征数据集
cfg.modelPath             = fullfile(cfg.outputDir, 'model.mat');              % 训练好的模型
cfg.segmentTemplatePath   = fullfile(cfg.outputDir, 'segment_template.xlsx');  % 分割窗口模板
cfg.hiddenDetailPath      = fullfile(cfg.outputDir, 'hidden_detail.xlsx');     % 隐藏集逐段预测
cfg.hiddenFileSummaryPath = fullfile(cfg.outputDir, 'hidden_file_summary.xlsx'); % 隐藏集文件级汇总
cfg.hiddenSubmitPath      = fullfile(cfg.outputDir, 'Pred_Labels.csv');        % 提交文件

%% ---- 预处理 ----
cfg.notchBaseHz = 50;    % 陷波基频 (Hz), 去除工频干扰
cfg.highpassHz  = 20;    % 高通截止频率 (Hz), 去除低频漂移和心电干扰
cfg.lowpassHz   = 500;   % 低通截止频率 (Hz), 保留肌电有效频段, 抑制高频噪声

%% ---- 动作分割 (自适应阈值) ----
cfg.useSegmentation = true;   % true=窗口切割, false=整文件作为一个样本 (快速验证用)

cfg.rmsWindow_s     = 0.15;   % RMS滑动窗宽度(s), ↑平滑 ↓时间分辨率
cfg.rmsStep_s       = 0.01;   % RMS滑动步长(s), ↓网格更密但计算量大
cfg.energySmooth_s  = 0.12;   % 能量曲线平滑窗宽(s), ↑曲线更平滑减少碎片
cfg.thresholdRatio  = 0.20;   % 自适应阈值 = 中位数 + 比例×(P95-中位数), ↓检出更多弱激活
cfg.minGap_s        = 0.20;   % 填小间隙(s), 短于此值的RMS掉落被合并
cfg.minDuration_s   = 0.35;   % 最短动作时长(s), 短于此的片段被丢弃 (滤噪声)
cfg.prePad_s        = 0.15;   % 窗口前填充(s), 捕获动作起始的低幅值信号
cfg.postPad_s       = 0.20;   % 窗口后填充(s), 捕获动作收尾的低幅值信号
cfg.sameActionGap_s = 0.75;   % 同动作双峰合并间距(s), 弯举的向心/离心双峰小于此值则合并

%% ---- 训练 ----
cfg.testRatio   = 0.3;   % 留出测试比例 (0.3 = 70%训练, 30%测试)
cfg.rfNumTrees  = 100;   % 随机森林树的数量, ↑更稳定但更慢, 样本少时宜少(50)
cfg.rfMinLeafSize = 5;   % 随机森林叶节点最少样本, ↑防过拟合, ↓可能欠拟合
end
