function [segments, info] = segment_actions(emg1, emg2, cfg, actionLabel)
% Segment EMG actions using adaptive-threshold on channel-normalized energy.
%
% Each channel's RMS is normalized by its own P95 before combination,
% so both channels contribute equally regardless of absolute amplitude.
% energy = sqrt((rms1/P95_1)^2 + (rms2/P95_2)^2)
%
% smooth → adaptive threshold → close small gaps → min duration → pad
%
% actionLabel is optional (accepted for backward compatibility, not used).

if nargin < 4
    actionLabel = [];  %#ok<NASGU>
end

% --- sliding-window RMS ---
winLen = round(cfg.rmsWindow_s * cfg.fs);
stepLen = round(cfg.rmsStep_s * cfg.fs);
numFrames = floor((length(emg1) - winLen) / stepLen) + 1;
tFrame = ((0:numFrames-1)' * stepLen + winLen / 2) / cfg.fs;

rms1 = zeros(numFrames, 1);
rms2 = zeros(numFrames, 1);
for i = 1:numFrames
    idx = (i-1) * stepLen + 1;
    rms1(i) = sqrt(mean(emg1(idx:idx+winLen-1).^2));
    rms2(i) = sqrt(mean(emg2(idx:idx+winLen-1).^2));
end

% --- channel-normalized energy ---
% 各通道除以自身P95, 两通道等权贡献
p95_1 = prctile(rms1, 95);
p95_2 = prctile(rms2, 95);
rms1Norm = rms1 / max(p95_1, eps);
rms2Norm = rms2 / max(p95_2, eps);
energy = sqrt(rms1Norm.^2 + rms2Norm.^2);

% --- adaptive threshold ---
smoothFrames = max(3, round(cfg.energySmooth_s * cfg.fs / stepLen));
energySmooth = movmean(energy, smoothFrames);

baseLevel = median(energySmooth);
activeLevel = prctile(energySmooth, 95);
dynamicRange = activeLevel - baseLevel;
threshold = baseLevel + cfg.thresholdRatio * dynamicRange;

minGapFrames = round(cfg.minGap_s * cfg.fs / stepLen);
minDurFrames = round(cfg.minDuration_s * cfg.fs / stepLen);

% --- extract active windows ---
isActive = energySmooth > threshold;
isActive = close_small_gaps(isActive, minGapFrames);

edges = diff([false; isActive; false]);
starts = find(edges == 1);
ends = find(edges == -1) - 1;

% filter by minimum duration
keep = (ends - starts + 1) >= minDurFrames;
starts = starts(keep);
ends = ends(keep);

% --- pad & merge ---
prePad = round(cfg.prePad_s * cfg.fs / stepLen);
postPad = round(cfg.postPad_s * cfg.fs / stepLen);
starts = max(1, starts - prePad);
ends = min(numFrames, ends + postPad);
[starts, ends] = merge_overlapping_windows(starts, ends);

% --- merge double-peak within same action ---
sameActionGapFrames = round(cfg.sameActionGap_s * cfg.fs / stepLen);
if ~isempty(starts) && numel(starts) > 1
    mergedStarts = starts(1);
    mergedEnds = ends(1);
    for i = 2:numel(starts)
        gap = starts(i) - mergedEnds(end);
        if gap <= sameActionGapFrames
            mergedEnds(end) = max(mergedEnds(end), ends(i));
        else
            mergedStarts(end+1) = starts(i); %#ok<AGROW>
            mergedEnds(end+1) = ends(i);     %#ok<AGROW>
        end
    end
    starts = mergedStarts;
    ends = mergedEnds;
end

% --- convert frame indices to sample indices ---
segments = zeros(numel(starts), 2);
for i = 1:numel(starts)
    sampleStart = max(1, (starts(i)-1) * stepLen + 1);
    sampleEnd = min(length(emg1), (ends(i)-1) * stepLen + winLen);
    segments(i, :) = [sampleStart, sampleEnd];
end

% --- diagnostic info ---
info.tFrame = tFrame;
info.rms1 = rms1Norm;
info.rms2 = rms2Norm;
info.energy = energySmooth;
info.threshold = threshold;
info.energyName = 'sqrt((rms1/P95_1)^2 + (rms2/P95_2)^2)';
info.thresholdRatio = cfg.thresholdRatio;
info.methodUsed = 'adaptive_threshold_norm';
end
