function [prediction, confidence] = emg_predict_two_stage(test_file)
% EMG_PREDICT_TWO_STAGE 两阶段分类预测（v2增强版）
% 输入:
%   test_file - 待预测的 .mat 文件路径
% 输出:
%   prediction - 预测标签 (1=弯举, 2=锤式弯举, 3=推肩)
%   confidence  - 投票置信度 (0~1)

    % 加载模型（路径相对于脚本所在目录）
    script_dir = fileparts(mfilename('fullpath'));
    model_path = fullfile(script_dir, 'emg_classification_v2', 'models', 'two_stage_model_v2.mat');
    m = load(model_path);

    % 加载数据
    data = load(test_file);
    if isfield(data, 'data')
        signal = double(data.data);
    else
        error('无法找到信号变量');
    end
    fs = 2000;

    % 预处理
    signal_filt = emg_preprocess(signal, fs);

    % 动作段检测
    segments = emg_segment_simple(signal_filt, fs);

    if isempty(segments)
        fprintf('未检测到动作段！\n');
        prediction = 0;
        confidence = 0;
        return;
    end

    fprintf('检测到 %d 个动作段\n', length(segments));

    % 对每个动作段进行两阶段分类
    predictions = zeros(length(segments), 1);

    for i = 1:length(segments)
        features = emg_feature_extract(segments{i}, fs);

        % 第一层：三头肌RMS, RMS比值, 能量比值 (特征索引 40, 37, 38)
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

    % 统计结果
    fprintf('\n=== 预测结果 ===\n');
    action_names = {'弯举', '锤式弯举', '推肩'};
    counts = histcounts(predictions, 0.5:3.5);

    for i = 1:3
        fprintf('  %s: %d 次 (%.1f%%)\n', action_names{i}, counts(i), 100*counts(i)/length(predictions));
    end

    [~, final_idx] = max(counts);
    prediction = final_idx;
    confidence = counts(final_idx) / length(predictions);

    fprintf('\n最终判断: %s (置信度: %.1f%%)\n', action_names{prediction}, confidence*100);
end
