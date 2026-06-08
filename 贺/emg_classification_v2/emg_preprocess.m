function [signal_filt] = emg_preprocess(signal, fs)
% EMG_PREPROCESS 肌电信号预处理
% 输入:
%   signal - 原始EMG信号 (N×1 或 N×2)
%   fs - 采样率 (Hz)
% 输出:
%   signal_filt - 预处理后的信号

    % 确保信号是列向量或双列矩阵
    if size(signal, 1) < size(signal, 2)
        signal = signal';
    end

    % 1. 去除直流与低频漂移 (0.5 Hz 高通)
    signal_filt = highpass(signal, 0.5, fs);

    % 2. 去趋势
    if size(signal_filt, 2) == 1
        signal_filt = detrend(signal_filt);
    else
        signal_filt = detrend(signal_filt);
    end

    % 3. 去除心电干扰 (20 Hz 高通)
    signal_filt = highpass(signal_filt, 20, fs);

    % 4. 抑制高频噪声 (500 Hz 低通)
    signal_filt = lowpass(signal_filt, 500, fs);

    % 5. 陷波滤波 (50 Hz 工频干扰)
    BW = 3;
    Fnyq = fs / 2;
    f0_norm = 50 / Fnyq;
    bw_norm = BW / Fnyq;
    [num, den] = designNotchPeakIIR(Response="notch", ...
                                    CenterFrequency=f0_norm, ...
                                    Bandwidth=bw_norm, ...
                                    FilterOrder=2);
    if size(signal_filt, 2) == 1
        signal_filt = filtfilt(num, den, signal_filt);
    else
        signal_filt(:,1) = filtfilt(num, den, signal_filt(:,1));
        signal_filt(:,2) = filtfilt(num, den, signal_filt(:,2));
    end
end
