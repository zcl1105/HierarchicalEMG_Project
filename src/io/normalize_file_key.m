function key = normalize_file_key(fileName)
key = strings(size(fileName));

for i = 1:numel(fileName)
    value = char(fileName(i));
    [~, name, ext] = fileparts(value);
    key(i) = lower(string([name ext]));
end
end
