function hiddenFiles = collect_hidden_files(hiddenDirs)
hiddenFiles = {};

for d = 1:numel(hiddenDirs)
    files = recursive_mat_files(hiddenDirs{d});
    hiddenFiles = [hiddenFiles; files(:)]; %#ok<AGROW>
end

if ~isempty(hiddenFiles)
    hiddenFiles = unique(hiddenFiles);
end
end
