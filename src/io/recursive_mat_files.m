function files = recursive_mat_files(rootDir)
if ~exist(rootDir, 'dir')
    files = {};
    return;
end

pathList = strsplit(genpath(rootDir), pathsep);
files = {};
for i = 1:numel(pathList)
    if isempty(pathList{i})
        continue;
    end

    listing = dir(fullfile(pathList{i}, '*.mat'));
    files = [files; fullfile({listing.folder}', {listing.name}')]; %#ok<AGROW>
end
end
