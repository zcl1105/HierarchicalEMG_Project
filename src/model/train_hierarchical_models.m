function models = train_hierarchical_models(X, Y, cfg, groups)
% Train 2-stage hierarchical classifier with 70/30 train/test split.
%
% Stage 1 (推肩=3 vs 弯举类):
%   SVM(RBF) — 三头RMS + RMS比 + 能量比 (3维)
%
% Stage 2 (弯举=1 vs 锤式弯举=2):
%   Random Forest — 双通道+跨通道 (24维)
%   groups(optional): file paths for leave-one-file-out CV diagnostic

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
model1 = fitcsvm(X1TrainZ, Y1Train, ...
    'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
    'Standardize', false, 'ClassNames', [0 1]);

% --- Stage 2: BicepsCurl vs HammerCurl (Random Forest) ---
curlTrainIdx = (YTrain ~= 3);
X2Train = XTrain(curlTrainIdx, cfg.stage2FeatureIdx);
Y2Train = YTrain(curlTrainIdx);
[X2TrainZ, mu2, sigma2] = zscore_safe(X2Train);
model2 = struct();
model2.rf = TreeBagger(100, X2TrainZ, Y2Train, 'Method', 'classification');
model2.isRF = true;

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

if nargin >= 4 && ~isempty(groups)
    groups = string(groups(:));
    models.stage2GroupDiagnostics = compute_stage2_group_diagnostics(X, Y, groups, cfg);
    fprintf('Stage2 leave-file-out diagnostic: %.1f%% (%d/%d curl segments)\n', ...
        models.stage2GroupDiagnostics.Accuracy * 100, ...
        models.stage2GroupDiagnostics.NCorrect, ...
        models.stage2GroupDiagnostics.NTotal);
    fprintf('  Confusion [true rows 1/2, pred cols 1/2]: [%d %d; %d %d]\n', ...
        models.stage2GroupDiagnostics.Confusion(1, 1), ...
        models.stage2GroupDiagnostics.Confusion(1, 2), ...
        models.stage2GroupDiagnostics.Confusion(2, 1), ...
        models.stage2GroupDiagnostics.Confusion(2, 2));
end
end

function diag = compute_stage2_group_diagnostics(X, Y, groups, cfg)
curlIdxAll = Y ~= 3;
curlGroups = unique(groups(curlIdxAll));
YPred = [];
YTrue = [];

for i = 1:numel(curlGroups)
    testIdx = curlIdxAll & groups == curlGroups(i);
    trainIdx = curlIdxAll & groups ~= curlGroups(i);
    if numel(unique(Y(trainIdx))) < 2
        continue;
    end

    XTrain = X(trainIdx, cfg.stage2FeatureIdx);
    YTrain = Y(trainIdx);
    XTest = X(testIdx, cfg.stage2FeatureIdx);
    YTest = Y(testIdx);

    [XTrainZ, mu, sigma] = zscore_safe(XTrain);
    rfModel = TreeBagger(100, XTrainZ, YTrain, 'Method', 'classification');
    XTestZ = apply_zscore_safe(XTest, mu, sigma);
    predCell = predict(rfModel, XTestZ);
    pred = str2double(predCell);

    YPred = [YPred; pred(:)]; %#ok<AGROW>
    YTrue = [YTrue; YTest(:)]; %#ok<AGROW>
end

conf = zeros(2, 2);
for r = 1:2
    for c = 1:2
        conf(r, c) = sum(YTrue == r & YPred == c);
    end
end

diag = struct();
diag.Accuracy = mean(YPred == YTrue);
diag.NCorrect = sum(YPred == YTrue);
diag.NTotal = numel(YTrue);
diag.Confusion = conf;
end
