function [emg1, emg2, t] = preprocess_emg(rawData, cfg)
channel1 = rawData(:, 1);
channel2 = rawData(:, 2);

N = min(length(channel1), length(channel2));
if mod(N, 2) ~= 0
    N = N - 1;
end

x1 = channel1(1:N);
x2 = channel2(1:N);
t = (0:N-1)' / cfg.fs;

x1 = detrend(x1);
x2 = detrend(x2);

% notch at 50 Hz and harmonics
maxNotchHz = min(cfg.lowpassHz, cfg.fs / 2 - 10);
harmonicCount = floor(maxNotchHz / cfg.notchBaseHz);
for k = 1:harmonicCount
    x1 = my_notch(cfg.notchBaseHz * k, cfg.fs, x1);
    x2 = my_notch(cfg.notchBaseHz * k, cfg.fs, x2);
end

emg1 = lowpass(highpass(x1, cfg.highpassHz, cfg.fs), cfg.lowpassHz, cfg.fs);
emg2 = lowpass(highpass(x2, cfg.highpassHz, cfg.fs), cfg.lowpassHz, cfg.fs);
end
