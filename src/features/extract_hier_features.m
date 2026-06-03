function [features, segmentTable] = extract_hier_features(emg1, emg2, segments, cfg)
% Extract 9-dim feature vector per action segment.
%
% Features 1-5 (basic):
%   RMS1, RMS2, Ratio, XCorrCoef, XCorrLag_ms
% Features 6-9:
%   MF1, MF2, nWL1, nWL2  (nWL = WL/RMS, 幅值无关的波形复杂度)

features = zeros(size(segments, 1), 9);
startTimes = zeros(size(segments, 1), 1);
endTimes = zeros(size(segments, 1), 1);

maxLag = round(cfg.xcorrMaxLag_s * cfg.fs);
envWin = round(cfg.envelopeSmooth_s * cfg.fs);

for i = 1:size(segments, 1)
    idx1 = segments(i, 1);
    idx2 = segments(i, 2);
    seg1 = emg1(idx1:idx2);
    seg2 = emg2(idx1:idx2);

    % --- Feature 1-2: RMS ---
    rms1 = sqrt(mean(seg1.^2));
    rms2 = sqrt(mean(seg2.^2));

    % --- Feature 3: Ratio ---
    ratio = rms1 / (rms2 + eps);

    % --- Feature 4-5: Cross-correlation ---
    env1 = movmean(abs(seg1), envWin);
    env2 = movmean(abs(seg2), envWin);
    env1 = env1 - mean(env1);
    env2 = env2 - mean(env2);

    [xc, lags] = xcorr(env1, env2, maxLag, 'coeff');
    [~, bestIdx] = max(abs(xc));
    corrCoef = xc(bestIdx);
    lagMs = lags(bestIdx) / cfg.fs * 1000;

    % --- Feature 6-7: Mean Frequency ---
    M1 = length(seg1);
    Y1 = fft(seg1);
    P1 = abs(Y1(1:floor(M1/2)+1)).^2;
    f1 = (0:floor(M1/2))' * cfg.fs / M1;
    mf1 = sum(f1 .* P1) / (sum(P1) + eps);

    M2 = length(seg2);
    Y2 = fft(seg2);
    P2 = abs(Y2(1:floor(M2/2)+1)).^2;
    f2 = (0:floor(M2/2))' * cfg.fs / M2;
    mf2 = sum(f2 .* P2) / (sum(P2) + eps);

    % --- Feature 8-9: Normalized Waveform Length ---
    wl1 = sum(abs(diff(seg1)));
    wl2 = sum(abs(diff(seg2)));
    nWL1 = wl1 / (rms1 + eps);
    nWL2 = wl2 / (rms2 + eps);

    features(i, :) = [rms1, rms2, ratio, corrCoef, lagMs, mf1, mf2, nWL1, nWL2];
    startTimes(i) = (idx1 - 1) / cfg.fs;
    endTimes(i) = (idx2 - 1) / cfg.fs;
end

segmentTable = table(startTimes, endTimes, ...
    'VariableNames', {'StartTime_s','EndTime_s'});
end
