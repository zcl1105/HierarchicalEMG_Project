function [YPred, stage1Pred, stage2Pred, stage1Scores, stage2Scores] = predict_hierarchical(X, models)
% Hierarchical prediction.
%
% Stage 1: SVM — 推肩(=3) vs 弯举类
% Stage 2: RF/SVM — 弯举(=1) vs 锤式弯举(=2)

% --- Stage 1 ---
X1 = apply_zscore_safe(X(:, models.stage1FeatureIdx), models.mu1, models.sigma1);
[stage1Pred, stage1Scores] = predict(models.model1, X1);

YPred = zeros(size(stage1Pred));
YPred(stage1Pred == 1) = 3;  % ShoulderPress → label 3

% --- Stage 2 ---
stage2Pred = nan(size(stage1Pred));
stage2Scores = nan(numel(stage1Pred), 2);
curlIdx = (stage1Pred == 0);  % 非推肩 → 进入Stage2
if any(curlIdx)
    X2 = apply_zscore_safe(X(curlIdx, models.stage2FeatureIdx), ...
                           models.mu2, models.sigma2);
    if isfield(models.model2, 'isRF') && models.model2.isRF
        % TreeBagger: predict returns cell array
        [predCell, scores] = predict(models.model2.rf, X2);
        stage2Pred(curlIdx) = str2double(predCell);
    else
        [stage2Pred(curlIdx), scores] = predict(models.model2, X2);
    end
    stage2Scores(curlIdx, :) = scores;
    YPred(curlIdx) = stage2Pred(curlIdx);  % label 1 or 2
end
end
