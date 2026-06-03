function trainFiles = collect_training_files(trainDirs, classNames)
trainFiles = {};

for d = 1:numel(trainDirs)
    files = recursive_mat_files(trainDirs{d});
    for i = 1:numel(files)
        [label, ok] = label_from_emg_file(files{i});
        if ~ok
            continue;
        end

        if ~isempty(trainFiles) && any(strcmp(trainFiles(:, 1), files{i}))
            continue;
        end

        trainFiles(end+1, :) = {files{i}, label, classNames{label}}; %#ok<AGROW>
    end
end

if ~isempty(trainFiles)
    trainFiles = sortrows(trainFiles, 1);
end
end
