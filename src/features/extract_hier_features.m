function [features, segmentTable] = extract_hier_features(emg1, emg2, segments, cfg)
% Extract 40-dim feature vector per action segment.
%
% 每通道18维 × 2 + 跨通道4维 = 40维
%
% 单通道时域(12): RMS, MAV, VAR, WL, ZC, SSC, WAMP, 偏度, 峰度, 峰值因子, 脉冲因子, IEMG
% 单通道频域(6):  MF, MDF, PF, 频谱熵, 带宽, 低频能量比
% 跨通道(4):     RMS比值, 能量比值, 相关性, 三头肌RMS
%
% Stage 1 (推肩检测): F40,F37,F38 → 3维
% Stage 2 (弯举细分): 双通道时域+频域+跨通道 → 24维 (具体索引见 config)

nSegs = size(segments, 1);
features = zeros(nSegs, 40);
startTimes = zeros(nSegs, 1);
endTimes = zeros(nSegs, 1);

for i = 1:nSegs
    idx1 = segments(i, 1);
    idx2 = segments(i, 2);
    seg1 = emg1(idx1:idx2);  % 肱二头肌
    seg2 = emg2(idx1:idx2);  % 肱三头肌
    N = length(seg1);

    % ===== 单通道特征提取 =====
    feat_ch1 = extract_channel_features(seg1, N, cfg.fs);
    feat_ch2 = extract_channel_features(seg2, N, cfg.fs);

    % ===== 跨通道特征 =====
    rms1  = feat_ch1(1);
    rms2  = feat_ch2(1);
    mav1  = feat_ch1(2);
    mav2  = feat_ch2(2);

    rms_ratio   = rms1 / (rms2 + eps);         % F37: RMS比值
    energy_ch1  = sum(seg1.^2);
    energy_ch2  = sum(seg2.^2);
    energy_ratio = energy_ch1 / (energy_ch2 + eps);  % F38: 能量比值
    corr_val    = corr(seg1, seg2);             % F39: 通道相关性
    tri_rms     = rms2;                         % F40: 三头肌RMS

    features(i, :) = [feat_ch1, feat_ch2, rms_ratio, energy_ratio, corr_val, tri_rms];
    startTimes(i) = (idx1 - 1) / cfg.fs;
    endTimes(i) = (idx2 - 1) / cfg.fs;
end

segmentTable = table(startTimes, endTimes, ...
    'VariableNames', {'StartTime_s','EndTime_s'});
end

%% ===== 单通道18维特征 =====
function feat = extract_channel_features(seg, N, fs)
    % ---- 时域特征 (12个) ----
    rms_val  = sqrt(mean(seg.^2));           % 1. RMS
    mav_val  = mean(abs(seg));               % 2. MAV
    var_val  = var(seg);                     % 3. VAR
    wl_val   = sum(abs(diff(seg)));          % 4. WL

    % ZC (过零率, 带阈值)
    th_zc = 0.01 * max(abs(seg));
    zc_val = sum(abs(diff(sign(seg))) > 0 & abs(diff(seg)) > th_zc) / N;  % 5. ZC

    % SSC (斜率符号变化)
    dseg = diff(seg);
    ssc_val = sum((dseg(1:end-1) .* dseg(2:end)) < 0) / max(N-2, 1);  % 6. SSC

    % WAMP (Willison幅度)
    th_wamp = 0.02 * max(abs(seg));
    wamp_val = sum(abs(diff(seg)) > th_wamp) / max(N-1, 1);  % 7. WAMP

    skew_val  = skewness(seg);               % 8. 偏度
    kurt_val  = kurtosis(seg);               % 9. 峰度
    peak_val  = max(abs(seg));
    crest_val = peak_val / (rms_val + eps);  % 10. 峰值因子
    impul_val = peak_val / (mav_val + eps);  % 11. 脉冲因子
    iemg_val  = sum(abs(seg));               % 12. IEMG

    % ---- 频域特征 (6个) ----
    Y = fft(seg);
    nHalf = floor(N/2) + 1;
    P = abs(Y(1:nHalf)).^2;
    f_axis = (0:nHalf-1)' * fs / N;
    P_norm = P / (sum(P) + eps);

    % MF (均值频率)
    mf_val = sum(f_axis .* P_norm);          % 13. MF

    % MDF (中位频率)
    cumP = cumsum(P_norm);
    idx_mdf = find(cumP >= 0.5, 1, 'first');
    mdf_val = f_axis(idx_mdf);               % 14. MDF

    % PF (主频率)
    [~, idx_pf] = max(P);
    pf_val = f_axis(idx_pf);                 % 15. PF

    % SE (频谱熵)
    se_val = -sum(P_norm .* log(P_norm + eps));  % 16. SE

    % BW (90%功率带宽)
    cumP_full = cumsum(P);
    idx5  = find(cumP_full >= 0.05 * cumP_full(end), 1, 'first');
    idx95 = find(cumP_full >= 0.95 * cumP_full(end), 1, 'first');
    bw_val = f_axis(idx95) - f_axis(idx5);   % 17. BW

    % 低频能量比 (0-100Hz)
    low_idx = f_axis <= 100;
    low_ratio = sum(P(low_idx)) / (sum(P) + eps);  % 18. LowRatio

    feat = [rms_val, mav_val, var_val, wl_val, zc_val, ssc_val, wamp_val, ...
            skew_val, kurt_val, crest_val, impul_val, iemg_val, ...
            mf_val, mdf_val, pf_val, se_val, bw_val, low_ratio];
end
