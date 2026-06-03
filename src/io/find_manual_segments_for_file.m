function segments = find_manual_segments_for_file(manualSegments, fileName, fs)
segments = [];

if isempty(manualSegments)
    return;
end

fileKey = normalize_file_key(string(fileName));
idx = (manualSegments.SourceFileKey == fileKey);
if ~any(idx)
    return;
end

T = manualSegments(idx, :);
T = sortrows(T, 'StartTime_s');
segments = zeros(height(T), 2);

for i = 1:height(T)
    sampleStart = max(1, round(T.StartTime_s(i) * fs) + 1);
    sampleEnd = max(sampleStart, round(T.EndTime_s(i) * fs) + 1);
    segments(i, :) = [sampleStart, sampleEnd];
end
end
