%% 分段诊断脚本 - 可视化测试文件的分段情况
% 修改 file_id 来查看不同文件
clear; clc; close all;

file_id = 10;  % 改成你想看的文件编号 1-36

base_dir = fileparts(mfilename('fullpath'));
test_file = fullfile(base_dir, 'Group09', sprintf('%d.mat', file_id));

data = load(test_file);
signal = double(data.data);
fs = 2000;

fprintf('文件: %d.mat\n', file_id);
fprintf('信号大小: %d × %d\n', size(signal,1), size(signal,2));
fprintf('时长: %.2f 秒\n', size(signal,1)/fs);

% 预处理
signal_filt = emg_preprocess(signal, fs);

% 计算能量包络（和分段函数一样的参数）
window_size = round(0.1 * fs);
step_size = round(0.05 * fs);
N = size(signal_filt, 1);

num_windows = floor((N - window_size) / step_size) + 1;
rms_energy = zeros(num_windows, 1);
for i = 1:num_windows
    idx_s = (i-1)*step_size + 1;
    idx_e = idx_s + window_size - 1;
    seg = signal_filt(idx_s:idx_e, :);
    rms_energy(i) = sqrt(mean(seg(:,1).^2)) + sqrt(mean(seg(:,2).^2));
end

time_axis = ((1:num_windows)-1) * step_size / fs;

% 分段
segments = emg_segment_simple(signal_filt, fs);
fprintf('检测到 %d 个动作段\n\n', length(segments));
for i = 1:length(segments)
    fprintf('  段%d: %d个采样点 (%.2f秒)\n', i, size(segments{i},1), size(segments{i},1)/fs);
end

%% 绘图
figure('Position', [100 100 1200 600]);

% 原始信号
subplot(3,1,1);
plot((0:N-1)/fs, signal_filt(:,1), 'b', 'LineWidth', 0.5); hold on;
plot((0:N-1)/fs, signal_filt(:,2), 'r', 'LineWidth', 0.5);
title(sprintf('文件 %d.mat - 原始信号 (蓝:二头肌 红:三头肌)', file_id));
xlabel('时间 (秒)');
legend('通道1', '通道2');

% 能量包络 + 阈值
subplot(3,1,2);
plot(time_axis, rms_energy, 'k', 'LineWidth', 0.8); hold on;
baseline_85 = prctile(rms_energy, 85);
yline(baseline_85, 'r--', '85%分位阈值', 'LineWidth', 1.5);

% 标注检测到的段
for i = 1:length(segments)
    t_start = (find(signal_filt(:,1) == segments{i}(1,1), 1) - 1) / fs;
    t_end = t_start + size(segments{i},1)/fs;
    xline(t_start, 'g', 'LineWidth', 1.5);
    xline(t_end, 'm', 'LineWidth', 1.5);
end

title(sprintf('能量包络 (检测到%d段)', length(segments)));
xlabel('时间 (秒)');

% 能量直方图
subplot(3,1,3);
histogram(rms_energy, 100);
xline(baseline_85, 'r--', 'LineWidth', 1.5);
title('能量分布直方图');
xlabel('RMS能量');
ylabel('窗口数');

fprintf('\n请根据图形调整分段参数\n');
