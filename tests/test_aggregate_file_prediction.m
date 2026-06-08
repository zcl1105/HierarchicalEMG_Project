function test_aggregate_file_prediction()
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));

cfg = struct();
cfg.fileVoteMinMargin = 2;

% Clear majority should use segment votes.
[label, stats] = aggregate_file_prediction([2; 2; 2; 1; 3], [0; 0; 0; 0; 1], ...
    [0.1 0.9; 0.2 0.8; 0.3 0.7; 0.9 0.1; nan nan], cfg);
assert(label == 2);
assert(stats.VoteCount2 == 3);
assert(stats.DecisionRule == "majority_vote");

% Close curl vote should use mean Stage 2 scores.
[label, stats] = aggregate_file_prediction([1; 1; 2; 2], [0; 0; 0; 0], ...
    [0.4 0.6; 0.4 0.6; 0.5 0.5; 0.45 0.55], cfg);
assert(label == 2);
assert(stats.DecisionRule == "stage2_score_tiebreak");

fprintf('test_aggregate_file_prediction passed\n');
end
