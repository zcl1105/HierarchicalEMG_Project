function [starts, ends] = merge_overlapping_windows(starts, ends)
if isempty(starts)
    return;
end

newStarts = starts(1);
newEnds = ends(1);
for i = 2:numel(starts)
    if starts(i) <= newEnds(end)
        newEnds(end) = max(newEnds(end), ends(i));
    else
        newStarts(end+1, 1) = starts(i); %#ok<AGROW>
        newEnds(end+1, 1) = ends(i); %#ok<AGROW>
    end
end

starts = newStarts;
ends = newEnds;
end
