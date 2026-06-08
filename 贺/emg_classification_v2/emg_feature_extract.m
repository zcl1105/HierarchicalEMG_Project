function features = emg_feature_extract(signal, fs)
% EMG_FEATURE_extract 提取肌电信号特征（扩展版）
% 输入:
%   signal - 预处理后的EMG信号 (N×2), 第1列肱二头肌, 第2列肱三头肌
%   fs - 采样率 (Hz)
% 输出:
%   features - 特征向量 (1×32)
%     时域特征(18): RMS, MAV, VAR, WL, ZC, SSC, WAMP, 偏度, 峰度, 峰值因子, 脉冲因子, IEMG (双通道各9个)
%     频域特征(10): MF, MDF, PF, SE, 带宽, 低频能量比 (双通道各5个)
%     协同特征(4): RMS比值, 能量比值, 相关性, 三头RMS

    num_channels = size(signal, 2);
    features = [];

    for ch = 1:num_channels
        seg = signal(:, ch);
        N = length(seg);

        %% 时域特征
        % 1. RMS (均方根值)
        rms_val = sqrt(mean(seg.^2));

        % 2. MAV (平均绝对值)
        mav = mean(abs(seg));

        % 3. VAR (方差)
        var_val = var(seg);

        % 4. WL (波形长度)
        wl = sum(abs(diff(seg)));

        % 5. ZC (过零率)
        threshold_zc = 0.01 * max(abs(seg));
        zc = sum(abs(diff(sign(seg))) > 0 & abs(diff(seg)) > threshold_zc);

        % 6. SSC (斜率符号变化数)
        diff_seg = diff(seg);
        ssc = sum((diff_seg(1:end-1) .* diff_seg(2:end)) < 0);

        % 7. WAMP (Willison幅度)
        threshold_wamp = 0.02 * max(abs(seg));
        wamp = sum(abs(diff(seg)) > threshold_wamp);

        % 8. 偏度 (Skewness) - 波形不对称性
        skew_val = skewness(seg);

        % 9. 峰度 (Kurtosis) - 波形尖锐程度
        kurt_val = kurtosis(seg);

        % 10. 峰值因子 (Crest Factor) - 峰值/RMS
        peak_val = max(abs(seg));
        crest_factor = peak_val / (rms_val + eps);

        % 11. 脉冲因子 (Impulse Factor) - 峰值/MAV
        impulse_factor = peak_val / (mav + eps);

        % 12. IEMG (积分肌电)
        iemg = sum(abs(seg));

        %% 频域特征
        % FFT
        Y = fft(seg);
        nfft_half = floor(N/2) + 1;
        P = abs(Y(1:nfft_half)).^2;
        f_axis = (0:nfft_half-1)' * fs / N;
        P_norm = P / (sum(P) + eps);

        % 13. MF (均值频率)
        mf = sum(f_axis .* P_norm);

        % 14. MDF (中位频率)
        cumP = cumsum(P_norm);
        idx_mdf = find(cumP >= 0.5, 1, 'first');
        mdf = f_axis(idx_mdf);

        % 15. PF (主频率)
        [~, idx_pf] = max(P);
        pf = f_axis(idx_pf);

        % 16. SE (频谱熵)
        se = -sum(P_norm .* log(P_norm + eps));

        % 17. 频带宽度 (Bandwidth) - 90%功率带宽
        cumP_full = cumsum(P);
        idx_5 = find(cumP_full >= 0.05 * cumP_full(end), 1, 'first');
        idx_95 = find(cumP_full >= 0.95 * cumP_full(end), 1, 'first');
        bandwidth = f_axis(idx_95) - f_axis(idx_5);

        % 18. 低频能量比 (0-100Hz / 总能量)
        low_freq_idx = f_axis <= 100;
        low_freq_ratio = sum(P(low_freq_idx)) / (sum(P) + eps);

        % 合并当前通道特征 (18个)
        features = [features, rms_val, mav, var_val, wl, zc, ssc, wamp, ...
                    skew_val, kurt_val, crest_factor, impulse_factor, iemg, ...
                    mf, mdf, pf, se, bandwidth, low_freq_ratio];
    end

    %% 协同特征 (通道间关系)
    % 19. RMS比值 (肱二头肌/肱三头肌)
    rms_ch1 = features(1);   % 肱二头肌RMS
    rms_ch2 = features(19);  % 肱三头肌RMS
    rms_ratio = rms_ch1 / (rms_ch2 + eps);

    % 20. 能量比值
    energy_ch1 = sum(signal(:,1).^2);
    energy_ch2 = sum(signal(:,2).^2);
    energy_ratio = energy_ch1 / (energy_ch2 + eps);

    % 21. 通道相关性
    corr_val = corr(signal(:,1), signal(:,2));

    % 22. 三头肌RMS（features第19个是三头肌RMS）
    tri_rms = features(19);

    features = [features, rms_ratio, energy_ratio, corr_val, tri_rms];
end
