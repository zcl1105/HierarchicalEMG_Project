function cfg = project_config(projectRoot)
cfg.projectRoot = projectRoot;
cfg.fs = 2000;
cfg.randomSeed = 1;
cfg.showTrainPlots  = false;
cfg.showHiddenPlots = false;
cfg.forceRetrain = true;

% --- versioning ---
cfg.runTimestamp = datestr(now, 'yyyymmdd_HHMMSS');
cfg.outputDir = fullfile(projectRoot, 'outputs', cfg.runTimestamp);
cfg.latestLink = fullfile(projectRoot, 'outputs', 'LATEST.txt');

% --- labels & features ---
cfg.classNames = {'BicepsCurl','HammerCurl','ShoulderPress'};
% 40维特征: ch1时域(1:12) ch1频域(13:18) ch2时域(19:30) ch2频域(31:36) 跨通道(37:40)
cfg.stage1FeatureIdx = [40, 37, 38];   % Stage 1: 推肩检测 (三头RMS,RMS比值,能量比值)
cfg.stage2FeatureIdx = [1:18, 40];     % Stage 2: 弯举细分 (二头18全特征+三头RMS)

% --- paths ---
cfg.trainDirs = {fullfile(projectRoot, 'data', 'train')};
cfg.hiddenTestDirs = {fullfile(projectRoot, 'data', 'hidden_test')};
cfg.manualSegmentPath = fullfile(projectRoot, 'data', 'manual_segments.xlsx');

cfg.datasetPath         = fullfile(cfg.outputDir, 'dataset.xlsx');
cfg.modelPath           = fullfile(cfg.outputDir, 'model.mat');
cfg.segmentTemplatePath = fullfile(cfg.outputDir, 'segment_template.xlsx');
cfg.hiddenDetailPath    = fullfile(cfg.outputDir, 'hidden_detail.xlsx');
cfg.hiddenSubmitPath    = fullfile(cfg.outputDir, 'Pred_Labels.csv');

% --- preprocessing ---
cfg.notchBaseHz = 50;
cfg.highpassHz  = 20;
cfg.lowpassHz   = 500;

% --- segmentation ---
cfg.useSegmentation = true;
cfg.rmsWindow_s     = 0.15;
cfg.rmsStep_s       = 0.01;
cfg.energySmooth_s  = 0.12;
cfg.thresholdRatio  = 0.20;
cfg.minGap_s        = 0.20;
cfg.minDuration_s   = 0.35;
cfg.prePad_s        = 0.15;
cfg.postPad_s       = 0.20;
cfg.sameActionGap_s = 0.75;

% --- training ---
cfg.testRatio = 0.30;
end
