function rawData = load_emg_matrix(fileName)
S = load(fileName);
names = fieldnames(S);

rawData = [];
for i = 1:numel(names)
    value = S.(names{i});
    if isnumeric(value) && ismatrix(value) && size(value, 2) >= 2 && size(value, 1) > 1000
        rawData = double(value(:, 1:2));
        return;
    end
end

error('No Nx2 numeric EMG matrix found in %s.', fileName);
end
