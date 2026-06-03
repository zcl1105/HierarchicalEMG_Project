function [label, ok] = label_from_emg_file(fileName)
[~, baseName] = fileparts(fileName);
token = regexp(lower(baseName), '^emg(\d+)$', 'tokens', 'once');

label = NaN;
ok = false;
if isempty(token)
    return;
end

id = str2double(token{1});
if id >= 1 && id < 100
    label = 1;
elseif id >= 100 && id < 200
    label = 2;
elseif id >= 200 && id < 300
    label = 3;
else
    return;
end

ok = true;
end
