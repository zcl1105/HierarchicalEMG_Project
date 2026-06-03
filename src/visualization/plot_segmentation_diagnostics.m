function plot_segmentation_diagnostics(t, emg1, emg2, segments, info, labelName, cfg, predLabels)
% predLabels: optional, string array of predicted class names for each segment
figure('Name', ['Segmentation - ', labelName], 'Position', [100 100 1200 700]);

subplot(2, 1, 1);
plot(t, emg1, 'b'); hold on;
plot(t, emg2, 'r');
yl = ylim;
for i = 1:size(segments, 1)
    xs = (segments(i, 1)-1) / cfg.fs;
    xe = (segments(i, 2)-1) / cfg.fs;
    fill([xs xe xe xs], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.3 0.8 0.4], 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    if nargin >= 8 && ~isempty(predLabels) && i <= numel(predLabels)
        text((xs + xe) / 2, yl(2) * 0.95, predLabels(i), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.8 0.2 0.2], ...
            'BackgroundColor', [1 1 1 0.7]);
    end
end
title(['Filtered EMG and detected windows - ', labelName]);
xlabel('Time (s)');
ylabel('Amplitude');
legend('Ch1','Ch2','Action window');
grid on;

subplot(2, 1, 2);
plot(info.tFrame, info.rms1, 'b'); hold on;
plot(info.tFrame, info.rms2, 'r');
plot(info.tFrame, info.energy, 'k', 'LineWidth', 1.2);
yline(info.threshold, 'm--', 'Threshold');
title(sprintf('Normalized RMS: %s | %s | thr=%.2f', ...
    info.methodUsed, info.energyName, info.thresholdRatio));
xlabel('Time (s)');
ylabel('Normalized RMS');
legend('RMS1 (norm)', 'RMS2 (norm)', 'Energy', 'Threshold');
grid on;
end
