function plot_confusion_result(YTrue, YPred, classNames, acc)
if exist('confusionchart', 'file') == 2
    figure('Name','Hierarchical classifier confusion matrix');
    confusionchart(categorical(YTrue, [1 2 3], classNames), ...
        categorical(YPred, [1 2 3], classNames));
    title(sprintf('Hierarchical SVM Accuracy = %.2f%%', acc));
else
    disp('Confusion matrix:');
    disp(confusionmat(YTrue, YPred));
end
end
