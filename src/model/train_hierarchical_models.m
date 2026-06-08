function models = train_hierarchical_models(X, Y, cfg)
% Train 2-stage hierarchical LDA classifier with 70/30 train/test split.
%
% Stage 1 (推肩=3 vs 弯举类=1,2):
%   三头RMS + RMS比值 + 能量比值 (3维) → LDA
%
% Stage 2 (弯举=1 vs 锤式弯举=2):
%   二头肌全特征 + 三头RMS (19维) → LDA

cv = cvpartition(Y, 'HoldOut', cfg.testRatio);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTest = X(testIdx, :);
YTest = Y(testIdx);

fprintf('\nTrain: %d | Test: %d\n', length(YTrain), length(YTest));

% --- Stage 1: ShoulderPress vs not ---
X1Train = XTrain(:, cfg.stage1FeatureIdx);
Y1Train = double(YTrain == 3);
[X1TrainZ, mu1, sigma1] = zscore_safe(X1Train);
model1 = fitcdiscr(X1TrainZ, Y1Train, 'DiscrimType', 'linear');

% --- Stage 2: BicepsCurl vs HammerCurl ---
curlTrainIdx = (YTrain ~= 3);
X2Train = XTrain(curlTrainIdx, cfg.stage2FeatureIdx);
Y2Train = YTrain(curlTrainIdx);
[X2TrainZ, mu2, sigma2] = zscore_safe(X2Train);
model2 = fitcdiscr(X2TrainZ, Y2Train, 'DiscrimType', 'linear');

fprintf('Stage1: %d features | Stage2: %d features, %d curl train / %d curl test\n', ...
    length(cfg.stage1FeatureIdx), length(cfg.stage2FeatureIdx), length(Y2Train), sum(YTest ~= 3));

% --- pack ---
models.model1 = model1;
models.model2 = model2;
models.mu1 = mu1;
models.sigma1 = sigma1;
models.mu2 = mu2;
models.sigma2 = sigma2;
models.stage1FeatureIdx = cfg.stage1FeatureIdx;
models.stage2FeatureIdx = cfg.stage2FeatureIdx;
models.classNames = cfg.classNames;
models.XTrain = XTrain;
models.YTrain = YTrain;
models.XTest = XTest;
models.YTest = YTest;
end
