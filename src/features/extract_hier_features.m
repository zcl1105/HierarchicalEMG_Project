function [features, segmentTable] = extract_hier_features(emg1, emg2, segments, cfg)
% Extract 9-dim feature vector per action segment.
%
% Stage 1 (推肩 vs 弯举类, 1:3):
%   RMS2, Ratio, Ratio_MAV  — 肱三头肌激活 + 二三头比值
%
% Stage 2 (弯举 vs 锤式弯举, 4:9):
%   RMS1, ZC, SSC, MF, MDF, PF  — 二头肌幅值 + 时域/频域特征

features = zeros(size(segments, 1), 9);
startTimes = zeros(size(segments, 1), 1);
endTimes = zeros(size(segments, 1), 1);

for i = 1:size(segments, 1)
    idx1 = segments(i, 1);
    idx2 = segments(i, 2);
    seg1 = emg1(idx1:idx2);  % 肱二头肌
    seg2 = emg2(idx1:idx2);  % 肱三头肌
    N1 = length(seg1);

    % ===== Stage 1 特征: 推肩 vs 弯举类 =====

    rms1 = sqrt(mean(seg1.^2));
    rms2 = sqrt(mean(seg2.^2));
    mav1 = mean(abs(seg1));
    mav2 = mean(abs(seg2));

    fRms2     = rms2;                  % F1: 肱三头肌RMS — 推肩主靠三头发力
    fRatio    = rms1 / (rms2 + eps);   % F2: RMS比值 — 推肩时比值低,弯举时高
    fRatioMAV = mav1 / (mav2 + eps);   % F3: 能量比值 — 同上,MAV版本

    % ===== Stage 2 特征: 弯举 vs 锤式弯举 (仅二头肌) =====

    fRMS1 = rms1;                      % F4: 肱二头肌RMS

    % 过零率 ZC
    zc = sum(diff(sign(seg1 - mean(seg1))) ~= 0) / N1;  % F5

    % 斜率变化 SSC
    dseg = diff(seg1);
    ssc = sum(diff(sign(dseg)) ~= 0) / max(N1 - 2, 1);  % F6

    % FFT
    Y1 = fft(seg1);
    P1 = abs(Y1(1:floor(N1/2)+1)).^2;
    f1 = (0:floor(N1/2))' * cfg.fs / N1;
    totalP = sum(P1) + eps;

    % 均值频率 MF
    mf = sum(f1 .* P1) / totalP;       % F7

    % 中位频率 MDF
    cumP = cumsum(P1);
    mdfIdx = find(cumP >= cumP(end)/2, 1, 'first');
    mdf = f1(mdfIdx);                  % F8

    % 主频率 PF
    [~, pfIdx] = max(P1);
    pf = f1(pfIdx);                    % F9

    features(i, :) = [fRms2, fRatio, fRatioMAV, fRMS1, zc, ssc, mf, mdf, pf];
    startTimes(i) = (idx1 - 1) / cfg.fs;
    endTimes(i) = (idx2 - 1) / cfg.fs;
end

segmentTable = table(startTimes, endTimes, ...
    'VariableNames', {'StartTime_s','EndTime_s'});
end
