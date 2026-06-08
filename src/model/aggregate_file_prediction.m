function [fileLabel, stats] = aggregate_file_prediction(predLabels, stage1Pred, stage2Scores, cfg)
% Aggregate segment predictions into one file-level label.
%
% Majority vote is used by default. For close BicepsCurl/HammerCurl votes,
% average Stage 2 SVM scores are used as a tie-breaker.

predLabels = predLabels(:);
stage1Pred = stage1Pred(:);

if isempty(predLabels)
    fileLabel = NaN;
    stats = make_stats(0, 0, 0, NaN, NaN, "empty");
    return;
end

if ~isfield(cfg, 'fileVoteMinMargin')
    cfg.fileVoteMinMargin = 2;
end

voteCounts = zeros(1, 3);
for c = 1:3
    voteCounts(c) = sum(predLabels == c);
end

[~, majorityLabel] = max(voteCounts);
sortedVotes = sort(voteCounts, 'descend');
voteMargin = sortedVotes(1) - sortedVotes(2);
curlMargin = abs(voteCounts(1) - voteCounts(2));
curlIdx = stage1Pred == 0 & predLabels <= 2;

meanScore1 = NaN;
meanScore2 = NaN;
if any(curlIdx) && nargin >= 3 && ~isempty(stage2Scores)
    curlScores = stage2Scores(curlIdx, :);
    meanScore1 = mean(curlScores(:, 1), 'omitnan');
    meanScore2 = mean(curlScores(:, 2), 'omitnan');
end

if majorityLabel <= 2 && curlMargin < cfg.fileVoteMinMargin && any(curlIdx) && ...
        nargin >= 3 && ~isempty(stage2Scores)
    if meanScore1 > meanScore2
        fileLabel = 1;
    else
        fileLabel = 2;
    end
    stats = make_stats(voteCounts(1), voteCounts(2), voteCounts(3), ...
        meanScore1, meanScore2, "stage2_score_tiebreak");
    return;
end

fileLabel = majorityLabel;
if voteMargin == 0
    rule = "first_mode_tie";
else
    rule = "majority_vote";
end

stats = make_stats(voteCounts(1), voteCounts(2), voteCounts(3), ...
    meanScore1, meanScore2, rule);
end

function stats = make_stats(v1, v2, v3, meanScore1, meanScore2, rule)
stats = struct();
stats.VoteCount1 = v1;
stats.VoteCount2 = v2;
stats.VoteCount3 = v3;
stats.Stage2MeanScore1 = meanScore1;
stats.Stage2MeanScore2 = meanScore2;
stats.DecisionRule = rule;
stats.TopVotes = max([v1, v2, v3]);
end
