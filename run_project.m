clc;
clear;
close all;

%% ============================================================
%  基于双通道 sEMG 的层级动作分类
%  Stage 1: 推肩 vs 非推肩 (5维)
%  Stage 2: 弯举 vs 锤式弯举 (9维: +MF +WL)
%
%  标签: 1=弯举  2=锤式弯举  3=推肩
%% ============================================================

%% 1. 初始化
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(fullfile(projectRoot, 'config'));

cfg = project_config(projectRoot);
rng(cfg.randomSeed);

%% 2. 收集训练数据
trainFiles = collect_training_files(cfg.trainDirs, cfg.classNames);
if isempty(trainFiles)
    error('No training EMG .mat files found under data/train/.');
end
fprintf('=== Training files: %d ===\n', size(trainFiles, 1));

manualSegments = load_manual_segments(cfg.manualSegmentPath);
if isempty(manualSegments)
    fprintf('Segmentation: auto\n\n');
else
    fprintf('Segmentation: manual override (%d rows)\n\n', height(manualSegments));
end

%% 3. 查找最新缓存 / 完整训练
latestModel = find_latest_model(fullfile(projectRoot, 'outputs'));
if ~isempty(latestModel) && ~cfg.forceRetrain
    % --- 命中缓存：直接加载 ---
    loaded = load(latestModel.path, 'models');
    models = loaded.models;
    allTable = readtable(latestModel.datasetPath);
    fprintf('=== Loaded cached model: %s ===\n', latestModel.version);
    fprintf('  Model:   %s\n', latestModel.path);
    fprintf('  Dataset: %s\n\n', latestModel.datasetPath);
else
    % --- 重新训练 ---
    if cfg.forceRetrain
        fprintf('=== forceRetrain=true, rebuilding... ===\n\n');
    end
    if ~exist(cfg.outputDir, 'dir'), mkdir(cfg.outputDir); end

    % 预处理 + 分割 + 特征提取
    allTable = table();
    segmentTemplate = table();
    for i = 1:size(trainFiles, 1)
        fileName = trainFiles{i, 1};
        label = trainFiles{i, 2};
        labelName = trainFiles{i, 3};

        rawData = load_emg_matrix(fileName);
        [emg1, emg2, t] = preprocess_emg(rawData, cfg);

        if cfg.useSegmentation
            manualFileSegments = find_manual_segments_for_file(manualSegments, fileName, cfg.fs);
            if isempty(manualFileSegments)
                [segments, rmsInfo] = segment_actions(emg1, emg2, cfg, label);
                segmentSource = "Auto";
            else
                segments = manualFileSegments;
                rmsInfo = [];
                segmentSource = "Manual";
            end
        else
            segments = [1, length(emg1)];   % 整文件作为一个样本
            rmsInfo = [];
            segmentSource = "WholeFile";
        end

        [featureMatrix, segTab] = extract_hier_features(emg1, emg2, segments, cfg);
        n = size(featureMatrix, 1);

        T = array2table(featureMatrix, 'VariableNames', cfg.featureNames);
        T.Label = repmat(label, n, 1);
        T.ActionName = repmat(string(labelName), n, 1);
        T.SourceFile = repmat(string(fileName), n, 1);
        T.ActionIndex = (1:n)';
        T.StartTime_s = segTab.StartTime_s;
        T.EndTime_s   = segTab.EndTime_s;
        T.SegmentSource = repmat(segmentSource, n, 1);

        allTable = [allTable; T]; %#ok<AGROW>
        segmentTemplate = [segmentTemplate; make_segment_template_rows(T)]; %#ok<AGROW>

        [~, fname, ext] = fileparts(fileName);
        fprintf('  %-25s  %-15s  %d samples\n', [fname ext], labelName, n);

        if cfg.showTrainPlots && segmentSource == "Auto"
            % 注: 训练集绘图时模型尚未训练, 无法标注预测结果; 仅显示真实标签和分割窗口
            plot_segmentation_diagnostics(t, emg1, emg2, segments, rmsInfo, labelName, cfg);
        end
    end

    % 导出数据集
    allTable = movevars(allTable, ...
        {'Label','ActionName','SourceFile','ActionIndex', ...
         'StartTime_s','EndTime_s','SegmentSource'}, 'Before', 1);
    writetable(allTable, cfg.datasetPath);
    writetable(segmentTemplate, cfg.segmentTemplatePath);

    % 训练
    X = allTable{:, cfg.featureNames};
    Y = allTable.Label;
    models = train_hierarchical_models(X, Y, cfg);
    save(cfg.modelPath, 'models');

    % 写版本指针
    fid = fopen(cfg.latestLink, 'w');
    fprintf(fid, '%s', cfg.runTimestamp);
    fclose(fid);

    fprintf('\nVersion: %s\n', cfg.runTimestamp);
    fprintf('Dataset: %s  (%d samples)\n', cfg.datasetPath, height(allTable));
    fprintf('Model:   %s\n', cfg.modelPath);
end

%% 6. 测试集评估
[YPred, stage1Pred, stage2Pred] = predict_hierarchical(models.XTest, models);
fprintf('\n========== Test Set Evaluation ==========');
print_prediction_table(models.YTest, YPred, stage1Pred, stage2Pred, cfg.classNames);
plot_confusion_result(models.YTest, YPred, cfg.classNames, ...
    mean(YPred == models.YTest) * 100);

%% 7. 隐藏测试集预测
hiddenFiles = collect_hidden_files(cfg.hiddenTestDirs);
if isempty(hiddenFiles)
    fprintf('\nNo hidden .mat files found under data/hidden_test/.\n');
    return;
end

if ~exist(cfg.outputDir, 'dir'), mkdir(cfg.outputDir); end
fprintf('========== Hidden Test: %d files ==========\n', numel(hiddenFiles));
hiddenTable = table();
fileOverallLabels = zeros(numel(hiddenFiles), 1);

for i = 1:numel(hiddenFiles)
    fileName = hiddenFiles{i};
    rawData = load_emg_matrix(fileName);
    [emg1, emg2, t] = preprocess_emg(rawData, cfg);

    if cfg.useSegmentation
        manualFileSegments = find_manual_segments_for_file(manualSegments, fileName, cfg.fs);
        if isempty(manualFileSegments)
            [segments, rmsInfo] = segment_actions(emg1, emg2, cfg);
            segmentSource = "Auto";
        else
            segments = manualFileSegments;
            rmsInfo = [];
            segmentSource = "Manual";
        end
    else
        segments = [1, length(emg1)];   % 整文件作为一个样本
        rmsInfo = [];
        segmentSource = "WholeFile";
    end

    [featureMatrix, segTab] = extract_hier_features(emg1, emg2, segments, cfg);

    [hiddenPred, hiddenStage1, hiddenStage2] = predict_hierarchical(featureMatrix, models);
    n = size(featureMatrix, 1);

    % 综合分析: 取所有片段的平均特征向量做整体预测
    meanFeatures = mean(featureMatrix, 1);
    filePred = predict_hierarchical(meanFeatures, models);
    overallLabel = filePred;
    overallName = labels_to_names(overallLabel, cfg.classNames);
    fileOverallLabels(i) = overallLabel;

    if cfg.showHiddenPlots && segmentSource == "Auto"
        [~, fname, ext] = fileparts(fileName);
        predActionNames = labels_to_names(hiddenPred, cfg.classNames);
        plot_segmentation_diagnostics(t, emg1, emg2, segments, rmsInfo, ...
            sprintf('Hidden: %s%s', fname, ext), cfg, predActionNames);
    end

    T = array2table(featureMatrix, 'VariableNames', cfg.featureNames);
    T.SourceFile = repmat(string(fileName), n, 1);
    T.ActionIndex = (1:n)';
    T.StartTime_s = segTab.StartTime_s;
    T.EndTime_s   = segTab.EndTime_s;
    T.SegmentSource = repmat(segmentSource, n, 1);
    T.PredLabel  = hiddenPred;
    T.PredAction = labels_to_names(hiddenPred, cfg.classNames);
    T.Stage1_IsShoulder = hiddenStage1;
    T.Stage2_CurlType   = hiddenStage2;

    hiddenTable = [hiddenTable; T]; %#ok<AGROW>

    [~, fname, ext] = fileparts(fileName);
    fprintf('  %-25s  %2d seg  -->  %s\n', [fname ext], n, overallName);
end

% --- 汇总 ---
fprintf('\n--- Hidden Summary ---\n');
fileIDs = zeros(numel(hiddenFiles), 1);
for i = 1:numel(hiddenFiles)
    [~, fname, ~] = fileparts(hiddenFiles{i});
    fileIDs(i) = str2double(fname);
end
filePreds = fileOverallLabels;
[fileIDs, sortIdx] = sort(fileIDs);
filePreds = filePreds(sortIdx);

for c = 1:3
    ids = fileIDs(filePreds == c);
    if ~isempty(ids)
        fprintf('  %-15s (%2d): ', cfg.classNames{c}, numel(ids));
        fprintf('%d ', ids);
        fprintf('\n');
    end
end

% 导出
submitTable = table(fileIDs, filePreds, 'VariableNames', {'ID', 'Pred_Label'});
writetable(submitTable, cfg.hiddenSubmitPath);

hiddenTable = movevars(hiddenTable, ...
    {'SourceFile','ActionIndex','StartTime_s','EndTime_s', ...
     'SegmentSource','PredLabel','PredAction'}, 'Before', 1);
writetable(hiddenTable, cfg.hiddenDetailPath);
fprintf('\nSubmit:  %s\n', cfg.hiddenSubmitPath);
fprintf('Details: %s\n', cfg.hiddenDetailPath);

%% ============================================================
function templateRows = make_segment_template_rows(T)
templateRows = table();
templateRows.SourceFile = T.SourceFile;
templateRows.ActionIndex = T.ActionIndex;
templateRows.StartTime_s = T.StartTime_s;
templateRows.EndTime_s = T.EndTime_s;
templateRows.Label = T.Label;
templateRows.ActionName = T.ActionName;
end

function latest = find_latest_model(outputsRoot)
latest = [];
if ~exist(outputsRoot, 'dir'), return; end

dirs = dir(fullfile(outputsRoot, '2*'));  % timestamp folders start with '2'
for d = numel(dirs):-1:1
    if ~dirs(d).isdir, continue; end
    modelFile = fullfile(dirs(d).folder, dirs(d).name, 'model.mat');
    datasetFile = fullfile(dirs(d).folder, dirs(d).name, 'dataset.xlsx');
    if exist(modelFile, 'file') && exist(datasetFile, 'file')
        latest.path = modelFile;
        latest.datasetPath = datasetFile;
        latest.version = dirs(d).name;
        return;
    end
end
end
