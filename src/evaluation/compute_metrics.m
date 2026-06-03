function results = compute_metrics(YTrue, YPred, classNames)
% Compute classification metrics for multi-class evaluation.
%
% Output struct fields:
%   confusion   - NxN confusion matrix (true rows, predicted cols)
%   accuracy    - overall accuracy (0-1)
%   precision   - per-class precision
%   recall      - per-class recall
%   f1          - per-class F1 score
%   classNames  - class name strings
%   nCorrect    - per-class correct count
%   nTotal      - per-class total count

nClasses = numel(classNames);
confusion = zeros(nClasses);
for i = 1:nClasses
    for j = 1:nClasses
        confusion(i, j) = sum(YTrue == i & YPred == j);
    end
end

nTotal = sum(confusion, 2);
nCorrect = diag(confusion);
accuracy = sum(nCorrect) / sum(nTotal);

precision = zeros(nClasses, 1);
recall = zeros(nClasses, 1);
f1 = zeros(nClasses, 1);
for c = 1:nClasses
    tp = confusion(c, c);
    fp = sum(confusion(:, c)) - tp;
    fn = sum(confusion(c, :)) - tp;
    precision(c) = tp / max(tp + fp, 1);
    recall(c) = tp / max(tp + fn, 1);
    f1(c) = 2 * precision(c) * recall(c) / max(precision(c) + recall(c), eps);
end

results.confusion = confusion;
results.accuracy = accuracy;
results.precision = precision;
results.recall = recall;
results.f1 = f1;
results.classNames = classNames;
results.nCorrect = nCorrect;
results.nTotal = nTotal;
end
