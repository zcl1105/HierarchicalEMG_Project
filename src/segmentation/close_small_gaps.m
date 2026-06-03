function isActive = close_small_gaps(isActive, maxGapFrames)
edges = diff([true; isActive(:); true]);
gapStarts = find(edges == -1);
gapEnds = find(edges == 1) - 1;

for i = 1:numel(gapStarts)
    if gapEnds(i) - gapStarts(i) + 1 <= maxGapFrames
        isActive(gapStarts(i):gapEnds(i)) = true;
    end
end
end
