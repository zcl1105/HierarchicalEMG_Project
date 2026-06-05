function models = train_hierarchical_models(X, Y, cfg)
% Train 2-stage hierarchical classifier with 70/30 train/test split.
%
% Stage 1 (ShoulderPress=3 vs rest=1,2):
%   RMS2, Ratio, Ratio_MAV → "三头肌激活+比值" 判推肩
%
% Stage 2 (BicepsCurl=1 vs HammerCurl=2):
%   RMS1, ZC, SSC, MF, MDF, PF → "二头幅值+时频域" 分弯举类型
%
% cfg.classifier: 'lda' (线性判别) or 'svm' (RBF-SVM)

cv = cvpartition(Y, 'HoldOut', cfg.testRatio);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTest = X(testIdx, :);
YTest = Y(testIdx);

fprintf('\nTrain: %d | Test: %d | Classifier: %s\n', length(YTrain), length(YTest), upper(cfg.classifier));

% --- Stage 1: ShoulderPress vs not ---
X1Train = XTrain(:, cfg.stage1FeatureIdx);
Y1Train = double(YTrain == 3);
[X1TrainZ, mu1, sigma1] = zscore_safe(X1Train);

switch lower(cfg.classifier)
    case 'lda'
        model1 = fitcdiscr(X1TrainZ, Y1Train, 'DiscrimType', 'linear');
    case 'svm'
        model1 = fitcsvm(X1TrainZ, Y1Train, ...
            'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
            'Standardize', false, 'ClassNames', [0 1]);
end

% --- Stage 2: BicepsCurl vs HammerCurl ---
curlTrainIdx = (YTrain ~= 3);
X2Train = XTrain(curlTrainIdx, cfg.stage2FeatureIdx);
Y2Train = YTrain(curlTrainIdx);
[X2TrainZ, mu2, sigma2] = zscore_safe(X2Train);

switch lower(cfg.classifier)
    case 'lda'
        model2 = fitcdiscr(X2TrainZ, Y2Train, 'DiscrimType', 'linear');
    case 'svm'
        model2 = fitcsvm(X2TrainZ, Y2Train, ...
            'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
            'Standardize', false, 'ClassNames', [1 2]);
end

fprintf('Stage2: %d curl train / %d curl test\n', length(Y2Train), sum(YTest ~= 3));

% --- pack ---
models.model1 = model1;
models.model2 = model2;
models.mu1 = mu1;
models.sigma1 = sigma1;
models.mu2 = mu2;
models.sigma2 = sigma2;
models.stage1FeatureIdx = cfg.stage1FeatureIdx;
models.stage2FeatureIdx = cfg.stage2FeatureIdx;
models.featureNames = cfg.featureNames;
models.classNames = cfg.classNames;
models.XTrain = XTrain;
models.YTrain = YTrain;
models.XTest = XTest;
models.YTest = YTest;
end
