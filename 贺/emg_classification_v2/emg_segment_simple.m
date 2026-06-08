function [segments] = emg_segment_simple(signal, fs)
% EMG_SEGMENT_SIMPLE 简化版动作段检测（无可视化，用于批量处理）
% 输入:
%   signal - EMG信号 (N×2)
%   fs - 采样率 (Hz)
% 输出:
%   segments - 动作段列表 (cell array)

    % 参数设置
    window_size = round(0.1 * fs);  % 滑动窗口 0.1秒
    step_size = round(0.05 * fs);   % 步长 0.05秒 (50%重叠)
    threshold_percentile = 85;       % 检测阈值

    N = size(signal, 1);

    % 计算滑动窗口RMS (合并两通道)
    num_windows = floor((N - window_size) / step_size) + 1;
    rms_energy = zeros(num_windows, 1);

    for i = 1:num_windows
        idx_start = (i-1) * step_size + 1;
        idx_end = idx_start + window_size - 1;
        segment = signal(idx_start:idx_end, :);
        % 两通道RMS之和作为能量指标
        rms_energy(i) = sqrt(mean(segment(:,1).^2)) + sqrt(mean(segment(:,2).^2));
    end

    % 检测动作阈值
    baseline = prctile(rms_energy, threshold_percentile);
    is_active = rms_energy > baseline;

    % 找到动作段起始点
    diff_active = diff([0; is_active; 0]);
    onset_indices = find(diff_active > 0);
    offset_indices = find(diff_active < 0) - 1;

    % 合并相邻的活跃段（间隔太近的合并）
    min_gap = 10; % 最小间隔（窗口数，约0.5秒）
    merged_onsets = [];
    merged_offsets = [];

    if ~isempty(onset_indices)
        current_onset = onset_indices(1);
        current_offset = offset_indices(1);

        for i = 2:length(onset_indices)
            if onset_indices(i) - current_offset <= min_gap
                % 合并
                current_offset = offset_indices(i);
            else
                % 保存当前段，开始新段
                merged_onsets = [merged_onsets; current_onset];
                merged_offsets = [merged_offsets; current_offset];
                current_onset = onset_indices(i);
                current_offset = offset_indices(i);
            end
        end
        % 保存最后一段
        merged_onsets = [merged_onsets; current_onset];
        merged_offsets = [merged_offsets; current_offset];
    end

    % 按能量排序，只保留最强的段（每文件4-6次重复）
    if length(merged_onsets) > 6
        seg_energies = zeros(length(merged_onsets), 1);
        for i = 1:length(merged_onsets)
            seg_energies(i) = mean(rms_energy(merged_onsets(i):merged_offsets(i)));
        end
        [~, sort_idx] = sort(seg_energies, 'descend');
        keep = sort(sort_idx(1:6));
        merged_onsets = merged_onsets(keep);
        merged_offsets = merged_offsets(keep);
    end

    % 提取动作段
    num_actions = length(merged_onsets);
    segments = cell(num_actions, 1);

    for i = 1:num_actions
        onset_win = merged_onsets(i);
        offset_win = merged_offsets(i);

        % 转换为样本索引
        start_idx = (onset_win - 1) * step_size + 1;
        end_idx = min((offset_win - 1) * step_size + window_size, N);

        % 确保最小长度
        if end_idx - start_idx + 1 >= window_size/2
            segments{i} = signal(start_idx:end_idx, :);
        end
    end

    % 移除空单元格
    segments = segments(~cellfun('isempty', segments));
end
