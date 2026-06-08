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
model2.rf = TreeBagger(cfg.rfNumTrees, X2TrainZ, Y2Train, 'Method', 'classification', 'MinLeafSize', cfg.rfMinLeafSize);
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
% 5-fold cross-validation on Stage 2 (curl vs hammer curl).
% Each fold is a stratified split of all curl samples.

curlIdx = Y ~= 3;
XCurl = X(curlIdx, cfg.stage2FeatureIdx);
YCurl = Y(curlIdx);

cv = cvpartition(YCurl, 'KFold', 5);
YPred = zeros(size(YCurl));

for fold = 1:5
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    XTrain = XCurl(trainIdx, :);
    YTrain = YCurl(trainIdx);
    XTest  = XCurl(testIdx, :);

    [XTrainZ, mu, sigma] = zscore_safe(XTrain);
    rfModel = TreeBagger(cfg.rfNumTrees, XTrainZ, YTrain, ...
        'Method', 'classification', 'MinLeafSize', cfg.rfMinLeafSize);
    XTestZ = apply_zscore_safe(XTest, mu, sigma);
    predCell = predict(rfModel, XTestZ);
    YPred(testIdx) = str2double(predCell);
end

conf = confusionmat(YCurl, YPred, 'Order', [1 2]);

% per-file error breakdown
curlGroups = string(groups(curlIdx));
allFiles = unique(curlGroups);
fprintf('  Stage2 CV per-file errors:\n');
for f = 1:numel(allFiles)
    fileIdx = curlGroups == allFiles(f);
    fileY = YCurl(fileIdx);
    filePred = YPred(fileIdx);
    nErr = sum(fileY ~= filePred);
    if nErr > 0
        errs = '';
        for s = 1:sum(fileIdx)
            if fileY(s) ~= filePred(s)
                errs = [errs sprintf('%d→%d ', fileY(s), filePred(s))]; %#ok<AGROW>
            end
        end
        [~, fname, ext] = fileparts(allFiles(f));
        fprintf('    %s%s: %d/%d [%s]\n', fname, ext, nErr, sum(fileIdx), strtrim(errs));
    end
end

diag = struct();
diag.Accuracy = mean(YPred == YCurl);
diag.NCorrect = sum(YPred == YCurl);
diag.NTotal = numel(YCurl);
diag.Confusion = conf;
end
