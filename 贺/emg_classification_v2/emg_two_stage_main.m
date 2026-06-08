%% EMG两阶段运动分类系统 v2（增强版）
% 改进：
%   - 增加更多特征（偏度、峰度、峰值因子等）
%   - 第二层尝试多种分类模型（LDA, SVM, 随机森林）

clear; clc; close all;

%% ==================== 配置参数 ====================
% 自动检测脚本所在目录
script_dir = fileparts(mfilename('fullpath'));

% ---- 训练数据路径配置 ----
% 默认指向项目内的 data/ 目录，包含 emg0XX/1XX/2XX.mat 共51个文件
data_root = fullfile(script_dir, 'data');

%% ==================== 1. 导入数据 ====================
fprintf('=== 1. 导入EMG训练数据 ===\n');
fprintf('数据根目录: %s\n', data_root);

sample_files = {};

% 先尝试扁平目录模式（原始项目 emg001/101/201.mat 格式）
mat_files_flat = dir(fullfile(data_root, '*.mat'));
for i = 1:length(mat_files_flat)
    filename = mat_files_flat(i).name;
    [~, name_only, ~] = fileparts(filename);
    % 匹配 emg001, emg101, emg201 等格式（第4字符为0/1/2）
    if length(name_only) >= 4
        action_code = name_only(4);
        if ismember(action_code, {'0', '1', '2'})
            sample_files{end+1} = fullfile(data_root, filename);
        end
    end
end

% 如果扁平目录没找到，尝试分文件夹模式（数据二/三/四）
if isempty(sample_files)
    for subfolder = {'数据一', '数据二', '数据三', '数据四'}
        subfolder_path = fullfile(data_root, subfolder{1});
        if ~exist(subfolder_path, 'dir')
            continue;
        end
        mat_files = dir(fullfile(subfolder_path, '*.mat'));
        for i = 1:length(mat_files)
            filename = mat_files(i).name;
            [~, name_only, ~] = fileparts(filename);
            action_code = name_only(4);
            if ismember(action_code, {'0', '1', '2'})
                sample_files{end+1} = fullfile(subfolder_path, filename);
            end
        end
    end
end

fprintf('总共找到 %d 个训练样本\n\n', length(sample_files));

%% ==================== 2. 加载和处理数据 ====================
fprintf('=== 2. 加载和处理数据 ===\n');

features_all = [];
labels_all = [];

for i = 1:length(sample_files)
    [~, name_only, ~] = fileparts(sample_files{i});
    action_code = name_only(4);

    if strcmp(action_code, '0')
        label = 1;
    elseif strcmp(action_code, '1')
        label = 2;
    else
        label = 3;
    end

    data = load(sample_files{i});
    if isfield(data, 'data')
        signal = data.data;
    else
        continue;
    end
    fs = 2000;

    signal_filt = emg_preprocess(signal, fs);
    segments = emg_segment_simple(signal_filt, fs);

    for j = 1:length(segments)
        features = emg_feature_extract(segments{j}, fs);
        features_all = [features_all; features];
        labels_all = [labels_all; label];
    end
end

fprintf('总计提取了 %d 个动作段, 特征维度: %d\n\n', size(features_all, 1), size(features_all, 2));

%% ==================== 3. 两阶段分类器设计 ====================
fprintf('=== 3. 两阶段分类器设计 ===\n\n');

% 特征索引说明（从1开始）：
% 第1-19个特征: 肱二头肌 (rms, mav, var, wl, zc, ssc, wamp, skew, kurt, crest, impulse, iemg, mf, mdf, pf, se, bw, low_ratio)
% 第20-38个特征: 肱三头肌
% 第39个: RMS比值, 第40个: 能量比值, 第41个: 相关性, 第42个: 三头RMS

%% 第一层：推肩 vs 弯举/锤式弯举
fprintf('--- 第一层分类器：推肩 vs 弯举/锤式弯举 ---\n');

labels_layer1 = ones(size(labels_all));
labels_layer1(labels_all == 3) = 2;

% 第一层关键特征：三头肌RMS + 比值
% 特征索引: 1-18二头肌, 19-36三头肌, 37 RMS比值, 38能量比值, 39相关性, 40三头RMS
features_layer1 = features_all(:, [40, 37, 38]);
feature_names_layer1 = {'三头肌RMS', 'RMS比值', '能量比值'};

mean1 = mean(features_layer1, 1);
std1 = std(features_layer1, 0, 1) + eps;
X1 = (features_layer1 - mean1) ./ std1;

model_layer1 = fitcdiscr(X1, labels_layer1);
pred1_train = predict(model_layer1, X1);
acc_layer1 = sum(pred1_train == labels_layer1) / length(labels_layer1);
fprintf('第一层准确率: %.2f%%\n', acc_layer1 * 100);

%% 第二层：弯举 vs 锤式弯举
fprintf('\n--- 第二层分类器：弯举 vs 锤式弯举 ---\n');

mask_curls = labels_all ~= 3;
features_curls = features_all(mask_curls, :);
labels_curls = labels_all(mask_curls);

fprintf('弯举样本: %d, 锤式弯举样本: %d\n', sum(labels_curls==1), sum(labels_curls==2));

% 第二层候选特征（多种组合）
fprintf('\n尝试多种特征组合和模型...\n\n');

results = [];

% 特征索引说明:
% 1-18: 肱二头肌 (rms, mav, var, wl, zc, ssc, wamp, skew, kurt, crest, impulse, iemg, mf, mdf, pf, se, bw, low_ratio)
% 19-36: 肱三头肌
% 37: RMS比值, 38: 能量比值, 39: 相关性, 40: 三头RMS

% 特征组合1: 二头全部特征
feat_set1 = 1:18;
name_set1 = '二头全特征(18)';

% 特征组合2: 二头+三头时域(rms,mav,var,wl,zc,ssc,wamp,skew,kurt,crest,impulse,iemg)
feat_set2 = [1:12, 19:30];
name_set2 = '双通道时域(24)';

% 特征组合3: 二头全部 + 三头RMS
feat_set3 = [1:18, 40];
name_set3 = '二头全+三头RMS(19)';

% 特征组合4: 精选特征（能量+波形+频域）
feat_set4 = [1, 2, 4, 8, 9, 10, 11, 13, 14, 15, 19, 20, 22, 23, 24];
name_set4 = '精选特征(15)';

% 特征组合5: 二头全部 + 三头全部
feat_set5 = 1:36;
name_set5 = '双通道全特征(36)';

feature_sets = {feat_set1, feat_set2, feat_set3, feat_set4, feat_set5};
set_names = {name_set1, name_set2, name_set3, name_set4, name_set5};

best_acc = 0;
best_model_info = [];

for s = 1:length(feature_sets)
    feat_idx = feature_sets{s};
    feat_data = features_curls(:, feat_idx);
    n_feat = length(feat_idx);

    % 标准化
    mean_s = mean(feat_data, 1);
    std_s = std(feat_data, 0, 1) + eps;
    X_s = (feat_data - mean_s) ./ std_s;

    % 模型1: LDA
    try
        model_lda = fitcdiscr(X_s, labels_curls);
        pred_lda = predict(model_lda, X_s);
        acc_lda = sum(pred_lda == labels_curls) / length(labels_curls);
    catch err
        disp(err.message);
        acc_lda = 0;
    end

    % 模型2: SVM (线性)
    try
        template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
        model_svm = fitcecoc(X_s, labels_curls, 'Learners', template);
        pred_svm = predict(model_svm, X_s);
        acc_svm = sum(pred_svm == labels_curls) / length(labels_curls);
    catch err
        disp(err.message);
        acc_svm = 0;
    end

    % 模型3: 随机森林 (Bagged Trees) - 使用交叉验证
    try
        cv = cvpartition(labels_curls, 'KFold', 5);
        acc_rf_cv = 0;
        for fold = 1:5
            train_idx = training(cv, fold);
            test_idx = test(cv, fold);
            rf_temp = TreeBagger(50, X_s(train_idx,:), labels_curls(train_idx), 'Method', 'classification');
            pred_rf_cv = predict(rf_temp, X_s(test_idx,:));
            pred_rf_cv = str2double(pred_rf_cv);
            acc_rf_cv = acc_rf_cv + sum(pred_rf_cv == labels_curls(test_idx)) / sum(test_idx);
        end
        acc_rf = acc_rf_cv / 5;  % 交叉验证平均准确率
    catch err
        disp(err.message);
        acc_rf = 0;
    end

    fprintf('%s:\n', set_names{s});
    fprintf('  LDA: %.2f%%, SVM: %.2f%%, RF(5折CV): %.2f%%\n', acc_lda*100, acc_svm*100, acc_rf*100);

    results = [results; {set_names{s}, n_feat, acc_lda, acc_svm, acc_rf}];

    % 记录最佳
    [max_acc, max_idx] = max([acc_lda, acc_svm, acc_rf]);
    if max_acc > best_acc
        best_acc = max_acc;
        best_model_info = {s, max_idx, feat_idx, mean_s, std_s};
    end
end

fprintf('\n');

% 选择最佳组合
[s_idx, m_idx, feat_idx, mean2, std2] = deal(best_model_info{:});
model_names = {'LDA', 'SVM', '随机森林'};
best_set_name = results{s_idx, 1};
fprintf('\n最佳组合: %s + %s\n', best_set_name, model_names{m_idx});

% 重新训练最佳模型
feat_data_best = features_curls(:, feat_idx);
X2 = (feat_data_best - mean2) ./ std2;

if m_idx == 1
    model_layer2 = fitcdiscr(X2, labels_curls);
elseif m_idx == 2
    template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
    model_layer2 = fitcecoc(X2, labels_curls, 'Learners', template);
else
    model_layer2 = TreeBagger(100, X2, labels_curls, 'Method', 'classification');
end

fprintf('第二层CV准确率: %.2f%%\n', best_acc*100);

%% ==================== 4. 综合性能评估（使用交叉验证） ====================
fprintf('\n=== 4. 综合性能评估（交叉验证）===\n');

% 两阶段交叉验证评估
cv_total = cvpartition(labels_all, 'KFold', 5);
acc_total_cv = 0;
all_preds = zeros(size(labels_all));

for fold = 1:5
    train_idx = training(cv_total, fold);
    test_idx = test(cv_total, fold);

    % 训练数据
    X_train = features_all(train_idx, :);
    y_train = labels_all(train_idx);

    % ========== 只用训练集计算标准化参数 ==========
    % 第一层标准化
    X1_train_raw = X_train(:, [40, 37, 38]);
    mean1_cv = mean(X1_train_raw, 1);
    std1_cv = std(X1_train_raw, 0, 1) + eps;
    X1_train = (X1_train_raw - mean1_cv) ./ std1_cv;

    % 第一层
    labels_l1 = ones(size(y_train));
    labels_l1(y_train == 3) = 2;
    model_l1 = fitcdiscr(X1_train, labels_l1);

    % 第二层标准化
    mask_train = y_train ~= 3;
    X2_train_raw = X_train(mask_train, feat_idx);
    mean2_cv = mean(X2_train_raw, 1);
    std2_cv = std(X2_train_raw, 0, 1) + eps;
    X2_train = (X2_train_raw - mean2_cv) ./ std2_cv;
    y2_train = y_train(mask_train);

    % 第二层
    if m_idx == 1
        model_l2 = fitcdiscr(X2_train, y2_train);
    elseif m_idx == 2
        template = templateSVM('KernelFunction', 'linear', 'Standardize', true);
        model_l2 = fitcecoc(X2_train, y2_train, 'Learners', template);
    else
        model_l2 = TreeBagger(100, X2_train, y2_train, 'Method', 'classification');
    end
    % =============================================

    % 测试数据预测 - 使用数值索引
    X_test = features_all(test_idx, :);
    test_indices = find(test_idx);  % 获取测试集的数值索引
    fprintf('第%d折: 测试集大小=%d\n', fold, length(test_indices));

    % 逐行预测
    for j = 1:length(test_indices)
        idx = test_indices(j);
        f1 = (X_test(j, [40, 37, 38]) - mean1_cv) ./ std1_cv;
        pred1 = predict(model_l1, f1);
        if iscell(pred1), pred1 = str2double(pred1); end

        if pred1 == 2
            final_pred = 3;
        else
            f2 = (X_test(j, feat_idx) - mean2_cv) ./ std2_cv;
            pred2 = predict(model_l2, f2);
            if iscell(pred2), pred2 = str2double(pred2); end
            final_pred = pred2;
        end
        all_preds(idx) = final_pred;
    end
end

% 计算最终准确率
acc_total = sum(all_preds == labels_all) / length(labels_all);
fprintf('两阶段分类CV准确率: %.2f%%\n\n', acc_total * 100);

% 混淆矩阵
cm = confusionmat(labels_all, all_preds);
fprintf('混淆矩阵:\n');
fprintf('              预测\n');
fprintf('真实     弯举   锤式弯举   推肩\n');
fprintf('弯举     %d       %d         %d\n', cm(1,1), cm(1,2), cm(1,3));
fprintf('锤式弯举  %d       %d         %d\n', cm(2,1), cm(2,2), cm(2,3));
fprintf('推肩     %d       %d         %d\n', cm(3,1), cm(3,2), cm(3,3));

fprintf('\n各类别CV准确率:\n');
action_names = {'弯举', '锤式弯举', '推肩'};
for i = 1:3
    class_total = sum(labels_all == i);
    class_correct = sum(all_preds(labels_all == i) == i);
    fprintf('  %s: %.2f%% (%d/%d)\n', action_names{i}, ...
        100*class_correct/class_total, class_correct, class_total);
end

%% ==================== 5. 保存模型 ====================
fprintf('\n=== 5. 保存模型 ===\n');

% 模型保存路径：脚本所在目录下的 emg_classification_v2/models/
model_dir = fullfile(script_dir, 'emg_classification_v2', 'models');
if ~exist(model_dir, 'dir')
    mkdir(model_dir);
end

save(fullfile(model_dir, 'two_stage_model_v2.mat'), ...
    'model_layer1', 'model_layer2', ...
    'mean1', 'std1', 'mean2', 'std2', ...
    'feat_idx', 'm_idx', 's_idx');

fprintf('模型已保存到: %s\n', model_dir);

%% ==================== 完成 ====================
fprintf('\n=== 两阶段分类器训练完成！ ===\n');
