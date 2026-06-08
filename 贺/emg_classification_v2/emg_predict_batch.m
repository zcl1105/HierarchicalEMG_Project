function emg_predict_batch()
% EMG_PREDICT_BATCH 批量预测Group09隐藏测试集并输出CSV

    % 路径配置
    base_dir = fileparts(mfilename('fullpath'));
    model_file = fullfile(base_dir, 'emg_classification_v2', 'models', 'two_stage_model_v2.mat');
    test_dir = fullfile(base_dir, 'Group09');
    output_file = fullfile(test_dir, 'Pred_Labels_Group09_Submit1.csv');

    % 加载模型（用结构体避免变量名与内置函数 mean2 冲突）
    m = load(model_file);

    % 获取所有测试文件，按数字排序
    file_list = dir(fullfile(test_dir, '*.mat'));
    ids = zeros(length(file_list), 1);
    for i = 1:length(file_list)
        ids(i) = sscanf(file_list(i).name, '%d.mat');
    end
    [ids, sort_idx] = sort(ids);
    file_list = file_list(sort_idx);

    fprintf('共找到 %d 个测试文件\n\n', length(file_list));

    % 逐文件预测
    pred_labels = zeros(length(file_list), 1);
    fs = 2000;
    action_names = {'弯举', '锤式弯举', '推肩'};

    for k = 1:length(file_list)
        file_path = fullfile(test_dir, file_list(k).name);
        data = load(file_path);
        if isfield(data, 'data')
            signal = double(data.data);
        else
            fprintf('[ID %d] 无法找到信号变量，跳过\n', ids(k));
            pred_labels(k) = 0;
            continue;
        end

        % 预处理
        signal_filt = emg_preprocess(signal, fs);

        % 动作段检测
        segments = emg_segment_simple(signal_filt, fs);

        if isempty(segments)
            fprintf('[ID %d] 未检测到动作段！\n', ids(k));
            pred_labels(k) = 0;
            continue;
        end

        % 对每个动作段进行两阶段分类
        predictions = zeros(length(segments), 1);
        for i = 1:length(segments)
            features = emg_feature_extract(segments{i}, fs);

            % 第一层
            f1 = (features([40, 37, 38]) - m.mean1) ./ m.std1;
            [pred1, ~] = predict(m.model_layer1, f1);
            if iscell(pred1), pred1 = str2double(pred1); end

            if pred1 == 2
                predictions(i) = 3;
            else
                f2 = (features(m.feat_idx) - m.mean2) ./ m.std2;
                pred2 = predict(m.model_layer2, f2);
                if iscell(pred2), pred2 = str2double(pred2); end
                predictions(i) = pred2;
            end
        end

        % 投票决定最终预测
        counts = histcounts(predictions, 0.5:3.5);
        [~, final_idx] = max(counts);
        pred_labels(k) = final_idx;

        fprintf('[ID %2d] %s (%d段, 置信度%.0f%%)\n', ...
            ids(k), action_names{final_idx}, length(segments), ...
            100 * counts(final_idx) / length(predictions));
    end

    % 写入CSV
    T = table(ids, pred_labels, 'VariableNames', {'ID', 'Pred_Label'});
    writetable(T, output_file);

    % 统计
    fprintf('\n=== 批量预测完成 ===\n');
    for i = 1:3
        cnt = sum(pred_labels == i);
        fprintf('  标签%d (%s): %d 个文件\n', i, action_names{i}, cnt);
    end
    fprintf('\n结果已保存到: %s\n', output_file);
end
