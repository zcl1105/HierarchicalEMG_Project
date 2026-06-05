function cfg = project_config(projectRoot)
cfg.projectRoot = projectRoot;
cfg.fs = 2000;
cfg.randomSeed = 1;
cfg.showTrainPlots  = false;
cfg.showHiddenPlots = true;
cfg.forceRetrain = true;

% --- versioning ---
cfg.runTimestamp = datestr(now, 'yyyymmdd_HHMMSS');
cfg.outputDir = fullfile(projectRoot, 'outputs', cfg.runTimestamp);
cfg.latestLink = fullfile(projectRoot, 'outputs', 'LATEST.txt');

% --- labels & features ---
cfg.classNames = {'BicepsCurl','HammerCurl','ShoulderPress'};
cfg.featureNames = {'RMS2','Ratio','Ratio_MAV', ...
                    'RMS1','ZC','SSC','MF','MDF','PF'};
cfg.stage1FeatureIdx = 1:3;   % Stage 1: 推肩检测 (RMS2,Ratio,Ratio_MAV — 三头激活+比值)
cfg.stage2FeatureIdx = 4:9;   % Stage 2: 弯举细分 (RMS1,ZC,SSC,MF,MDF,PF — 二头幅值+时频域)

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
