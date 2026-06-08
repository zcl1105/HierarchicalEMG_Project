function [YPred, stage1Pred, stage2Pred] = predict_hierarchical(X, models)
% Hierarchical LDA prediction.
%
% Stage 1: 推肩(=3) vs 弯举类, uses models.stage1FeatureIdx
% Stage 2: 弯举(=1) vs 锤式弯举(=2), uses models.stage2FeatureIdx

% --- Stage 1 ---
X1 = apply_zscore_safe(X(:, models.stage1FeatureIdx), models.mu1, models.sigma1);
stage1Pred = predict(models.model1, X1);

YPred = zeros(size(stage1Pred));
YPred(stage1Pred == 1) = 3;  % ShoulderPress → label 3

% --- Stage 2 ---
stage2Pred = nan(size(stage1Pred));
curlIdx = (stage1Pred ~= 2);
if any(curlIdx)
    X2 = apply_zscore_safe(X(curlIdx, models.stage2FeatureIdx), ...
                           models.mu2, models.sigma2);
    stage2Pred(curlIdx) = predict(models.model2, X2);
    YPred(curlIdx) = stage2Pred(curlIdx);  % label 1 or 2
end
end
