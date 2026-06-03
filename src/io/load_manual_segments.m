function manualSegments = load_manual_segments(manualSegmentPath)
manualSegments = table();

if ~exist(manualSegmentPath, 'file')
    return;
end

manualSegments = readtable(manualSegmentPath, 'TextType', 'string');
requiredNames = ["SourceFile", "StartTime_s", "EndTime_s"];
for i = 1:numel(requiredNames)
    if ~ismember(requiredNames(i), string(manualSegments.Properties.VariableNames))
        error('Manual segment file must contain column: %s', requiredNames(i));
    end
end

manualSegments.SourceFileKey = normalize_file_key(manualSegments.SourceFile);
end
