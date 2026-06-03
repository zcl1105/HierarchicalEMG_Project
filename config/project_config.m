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
cfg.featureNames = {'RMS1','RMS2','Ratio','XCorrCoef','XCorrLag_ms', ...
                    'MF1','MF2','nWL1','nWL2'};
cfg.stage1FeatureIdx = 1:4;   % RMS1, RMS2, Ratio, XCorrCoef
cfg.stage2FeatureIdx = 1:9;   % 全部9维

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

% --- features ---
cfg.xcorrMaxLag_s    = 0.30;
cfg.envelopeSmooth_s = 0.05;

% --- training ---
cfg.testRatio = 0.30;
end
