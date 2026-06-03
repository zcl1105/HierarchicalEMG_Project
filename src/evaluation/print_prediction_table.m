function print_prediction_table(YTrue, YPred, stage1Pred, stage2Pred, classNames)
% Print clean evaluation summary: confusion matrix + per-class metrics + stage accuracy.

results = compute_metrics(YTrue, YPred, classNames);
nClasses = numel(classNames);

% Column widths
nameWidth = max(cellfun(@length, classNames)) + 2;
colWidth = 10;

% --- Confusion Matrix ---
fprintf('\n  Confusion Matrix (row=True, col=Pred):\n');
fprintf('  %-*s', nameWidth, '');
for j = 1:nClasses
    fprintf('%-*s', colWidth, classNames{j});
end
fprintf('%*s\n', colWidth, 'Total');
for i = 1:nClasses
    fprintf('  %-*s', nameWidth, classNames{i});
    for j = 1:nClasses
        fprintf('%-*d', colWidth, results.confusion(i, j));
    end
    fprintf('%-*d\n', colWidth, results.nTotal(i));
end

% --- Per-class Metrics ---
fprintf('\n  Per-class Metrics:\n');
fprintf('  %-*s  %10s  %10s  %10s  %10s\n', ...
    nameWidth, 'Class', 'Precision', 'Recall', 'F1', 'Accuracy');
sepLine = repmat('-', 1, nameWidth + 50);
fprintf('  %s\n', sepLine);
for c = 1:nClasses
    classAcc = results.nCorrect(c) / max(results.nTotal(c), 1) * 100;
    fprintf('  %-*s  %9.1f%%  %9.1f%%  %9.3f  %9.1f%%\n', ...
        nameWidth, classNames{c}, ...
        results.precision(c) * 100, ...
        results.recall(c) * 100, ...
        results.f1(c), ...
        classAcc);
end
fprintf('  %s\n', sepLine);
fprintf('  %-*s  %44.1f%%\n', nameWidth, 'OVERALL', results.accuracy * 100);

% --- Stage Accuracy ---
stage1Correct = (stage1Pred == double(YTrue == 3));
stage1Acc = mean(stage1Correct) * 100;
curlIdx = ~isnan(stage2Pred);
if any(curlIdx)
    stage2Correct = (stage2Pred(curlIdx) == YTrue(curlIdx));
    stage2Acc = mean(stage2Correct) * 100;
    nCurl = sum(curlIdx);
else
    stage2Acc = NaN;
    nCurl = 0;
end

fprintf('\n  Stage Accuracy:\n');
fprintf('  Stage1 (ShoulderPress vs Rest): %.1f%% (%d samples)\n', stage1Acc, numel(YTrue));
if ~isnan(stage2Acc)
    fprintf('  Stage2 (BicepsCurl vs HammerCurl): %.1f%% (%d samples)\n', stage2Acc, nCurl);
end
fprintf('\n');
end
