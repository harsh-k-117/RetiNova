% EVALUATEIDRIDGRADER
% Evaluate the frozen fold ensemble once on the official IDRiD test set.
% ALL decoder thresholds must have been learned from cross-validation/OOF
% predictions before this script is run.
%
% Decoder:
%   1) Ordinal expected-grade thresholds produce the base prediction.
%   2) Grade 4 is forced when:
%        - argmax probability is class 4, OR
%        - P(grade 4) >= learned grade4ProbThreshold
%
% No threshold fitting is performed on the official test set.

clear; clc;

projectDir = fileparts( ('fullpath'));

dataRoot = fullfile(projectDir, 'data', 'B. Disease Grading', ...
    'B. Disease Grading');

archive = fullfile(projectDir, 'data', 'B. Disease Grading.zip');

testImageDir = fullfile(dataRoot, ...
    '1. Original Images', 'b. Testing Set');

testCsv = fullfile(dataRoot, ...
    '2. Groundtruths', ...
    'b. IDRiD_Disease Grading_Testing Labels.csv');

modelDir = fullfile(projectDir, 'models', 'idrid_grade5');

% -------------------------------------------------------------------------
% Prepare data
% -------------------------------------------------------------------------
if ~isfolder(testImageDir) || isempty(dir(fullfile(testImageDir, '*.jpg')))
    unzip(archive, fullfile(projectDir, 'data', 'B. Disease Grading'));
end

if ~isfile(fullfile(modelDir, 'fold-1.mat'))
    error('Train the fold models first: %s', modelDir);
end

% -------------------------------------------------------------------------
% Discover folds
% -------------------------------------------------------------------------
foldPaths = {};

for fold = 1:10
    modelPath = fullfile(modelDir, sprintf('fold-%d.mat', fold));

    if ~isfile(modelPath)
        break
    end

    foldPaths{end+1} = modelPath; 
end

nFolds = numel(foldPaths);

fprintf('Evaluating %d-fold ensemble on official test\n', nFolds);

% -------------------------------------------------------------------------
% Load official test labels
% -------------------------------------------------------------------------
T = readtable(testCsv);

grades = double(T.RetinopathyGrade);
names = string(T.ImageName);

% -------------------------------------------------------------------------
% Load frozen decoder parameters learned ONLY from CV / OOF
% -------------------------------------------------------------------------
calibrationPath = fullfile(modelDir, 'cross_validation.mat');

if ~isfile(calibrationPath)
    error('Missing calibration file: %s', calibrationPath);
end

calibration = load(calibrationPath);

if ~isfield(calibration, 'thresholds')
    error('cross_validation.mat does not contain "thresholds".');
end

thresholds = double(calibration.thresholds(:)');

if numel(thresholds) ~= 4
    error('Expected 4 ordinal thresholds for grades 0..4.');
end

% Grade-4 probability threshold must come from OOF/CV.
if isfield(calibration, 'grade4ProbThreshold')
    grade4ProbThreshold = double(calibration.grade4ProbThreshold);
elseif isfield(calibration, 'grade4Threshold')
    % Backward-compatible name.
    grade4ProbThreshold = double(calibration.grade4Threshold);
else
    error(['cross_validation.mat is missing grade4ProbThreshold. ' ...
           'Fit this threshold on OOF predictions before evaluating the test set.']);
end

fprintf('Frozen ordinal thresholds: ');
fprintf('%.4f ', thresholds);
fprintf('\n');

fprintf('Frozen grade-4 probability threshold: %.4f\n', ...
    grade4ProbThreshold);

% -------------------------------------------------------------------------
% Ensemble prediction
% -------------------------------------------------------------------------
allScores = zeros(numel(names), 5, 'single');

for f = 1:nFolds

    S = load(foldPaths{f}, 'net', 'inputSize');

    hasSoftmax = any(arrayfun( ...
        @(layer) isa(layer, 'nnet.cnn.layer.SoftmaxLayer'), ...
        S.net.Layers));

    for k = 1:numel(names)

        imagePath = fullfile(testImageDir, names(k) + ".jpg");

        I = preprocessIDRiDImage(imagePath, S.inputSize);

        out = predict(S.net, dlarray(I, 'SSCB'));

        if ~hasSoftmax
            out = softmax(out);
        end

        scores = extractdata(out);
        scores = reshape(scores, 1, []);

        if numel(scores) < 5
            error('Network output has fewer than 5 classes.');
        end

        allScores(k, :) = ...
            allScores(k, :) + single(scores(1:5)) / nFolds;
    end

    fprintf('Scored test set with fold %d / %d\n', f, nFolds);
end

% -------------------------------------------------------------------------
% Base ordinal decoder
% -------------------------------------------------------------------------
expectedGrades = double(allScores) * (0:4)';

ordinalPredictions = zeros(size(expectedGrades));

for t = thresholds
    ordinalPredictions = ordinalPredictions + ...
        (expectedGrades > t);
end

% -------------------------------------------------------------------------
% Argmax decoder
% -------------------------------------------------------------------------
[~, argmaxIdx] = max(allScores, [], 2);

argmaxPredictions = argmaxIdx - 1;  % Convert MATLAB index -> grade 0..4

% -------------------------------------------------------------------------
% Hybrid decoder
% -------------------------------------------------------------------------
%
% Force grade 4 when:
%   A) class 4 is already argmax, OR
%   B) P(grade 4) exceeds the frozen OOF threshold.
%
% Otherwise retain the ordinal prediction.
%
% This is NOT sample-specific threshold tuning. The threshold was frozen
% before seeing the official test labels.
% -------------------------------------------------------------------------
predictions = ordinalPredictions;

forceGrade4 = ...
    (argmaxPredictions == 4) | ...
    (double(allScores(:, 5)) >= grade4ProbThreshold);

predictions(forceGrade4) = 4;

% -------------------------------------------------------------------------
% Confusion matrix
% -------------------------------------------------------------------------
confusion = confusionmat( ...
    grades, predictions, 'Order', 0:4);

% -------------------------------------------------------------------------
% Exact 5-class metrics
% -------------------------------------------------------------------------
accuracy = mean(predictions == grades);

recall = diag(confusion) ./ max(sum(confusion, 2), 1);

precision = diag(confusion) ./ max(sum(confusion, 1)', 1);

f1 = 2 .* precision .* recall ./ ...
    max(precision + recall, eps);

macroRecall = mean(recall);
macroF1 = mean(f1);

% -------------------------------------------------------------------------
% Quadratic Weighted Kappa
% -------------------------------------------------------------------------
n = sum(confusion(:));

[r, c] = ndgrid(0:4, 0:4);

w = (r - c).^2 / 16;

observed = ...
    sum(w(:) .* confusion(:)) / max(n, 1);

expectedCounts = ...
    sum(confusion, 2) * sum(confusion, 1) / max(n, 1);

expected = ...
    sum(w(:) .* expectedCounts(:)) / max(n, 1);

qwk = 1 - observed / max(expected, eps);

% -------------------------------------------------------------------------
% Grade-4 metrics
% -------------------------------------------------------------------------
trueGrade4 = grades == 4;
predGrade4 = predictions == 4;

grade4TP = sum(trueGrade4 & predGrade4);
grade4FN = sum(trueGrade4 & ~predGrade4);
grade4FP = sum(~trueGrade4 & predGrade4);
grade4TN = sum(~trueGrade4 & ~predGrade4);

grade4Recall = grade4TP / max(grade4TP + grade4FN, 1);

grade4Precision = grade4TP / max(grade4TP + grade4FP, 1);

grade4F1 = ...
    2 * grade4Precision * grade4Recall / ...
    max(grade4Precision + grade4Recall, eps);

% -------------------------------------------------------------------------
% Referable DR metrics
% -------------------------------------------------------------------------
%
% IDRiD grade convention:
%   0 = No DR
%   1 = Mild NPDR
%   2 = Moderate NPDR
%   3 = Severe NPDR
%   4 = Proliferative DR
%
% Referable DR = grades 2, 3, 4
% Non-referable = grades 0, 1
% -------------------------------------------------------------------------
trueReferable = grades >= 2;
predReferable = predictions >= 2;

refTP = sum(trueReferable & predReferable);
refFN = sum(trueReferable & ~predReferable);
refFP = sum(~trueReferable & predReferable);
refTN = sum(~trueReferable & ~predReferable);

referableSensitivity = ...
    refTP / max(refTP + refFN, 1);

referableSpecificity = ...
    refTN / max(refTN + refFP, 1);

referablePrecision = ...
    refTP / max(refTP + refFP, 1);

referableF1 = ...
    2 * referablePrecision * referableSensitivity / ...
    max(referablePrecision + referableSensitivity, eps);

% -------------------------------------------------------------------------
% Decoder comparison
% -------------------------------------------------------------------------
ordinalAccuracy = mean(ordinalPredictions == grades);

ordinalGrade4Recall = ...
    sum((grades == 4) & (ordinalPredictions == 4)) / ...
    max(sum(grades == 4), 1);

hybridGrade4Recall = grade4Recall;

fprintf('\n');
fprintf('============================================\n');
fprintf('OFFICIAL IDRiD TEST RESULTS\n');
fprintf('============================================\n');

fprintf('5-class accuracy       : %.4f\n', accuracy);
fprintf('QWK                    : %.4f\n', qwk);
fprintf('Macro recall           : %.4f\n', macroRecall);
fprintf('Macro F1               : %.4f\n', macroF1);

fprintf('\n');
fprintf('Grade 4 recall         : %.4f\n', grade4Recall);
fprintf('Grade 4 precision      : %.4f\n', grade4Precision);
fprintf('Grade 4 F1             : %.4f\n', grade4F1);
fprintf('Predicted grade 4      : %d\n', sum(predGrade4));
fprintf('True grade 4           : %d\n', sum(trueGrade4));

fprintf('\n');
fprintf('REFERABLE DR (2-4)\n');
fprintf('Sensitivity            : %.4f\n', referableSensitivity);
fprintf('Specificity            : %.4f\n', referableSpecificity);
fprintf('Precision              : %.4f\n', referablePrecision);
fprintf('F1                     : %.4f\n', referableF1);

fprintf('\n');
fprintf('DECODER COMPARISON\n');
fprintf('Ordinal accuracy       : %.4f\n', ordinalAccuracy);
fprintf('Ordinal grade-4 recall : %.4f\n', ordinalGrade4Recall);
fprintf('Hybrid grade-4 recall  : %.4f\n', hybridGrade4Recall);

fprintf('\n');
fprintf('Official test confusion matrix\n');
fprintf('(true rows, predicted columns)\n');

disp(array2table( ...
    confusion, ...
    'VariableNames', "Pred_" + string(0:4), ...
    'RowNames', "True_" + string(0:4)));

% -------------------------------------------------------------------------
% Save detailed predictions
% -------------------------------------------------------------------------
results = table( ...
    names, ...
    grades, ...
    ordinalPredictions, ...
    argmaxPredictions, ...
    predictions, ...
    expectedGrades, ...
    allScores(:,1), ...
    allScores(:,2), ...
    allScores(:,3), ...
    allScores(:,4), ...
    allScores(:,5), ...
    forceGrade4, ...
    'VariableNames', { ...
        'Image', ...
        'TrueGrade', ...
        'OrdinalPrediction', ...
        'ArgmaxPrediction', ...
        'PredictedGrade', ...
        'ExpectedGrade', ...
        'Score0', ...
        'Score1', ...
        'Score2', ...
        'Score3', ...
        'Score4', ...
        'ForcedGrade4'});

writetable( ...
    results, ...
    fullfile(modelDir, 'official_test_predictions.csv'));

% -------------------------------------------------------------------------
% Save report
% -------------------------------------------------------------------------
save( ...
    fullfile(modelDir, 'official_test_report.mat'), ...
    'confusion', ...
    'accuracy', ...
    'qwk', ...
    'macroRecall', ...
    'macroF1', ...
    'precision', ...
    'recall', ...
    'f1', ...
    'grade4Recall', ...
    'grade4Precision', ...
    'grade4F1', ...
    'referableSensitivity', ...
    'referableSpecificity', ...
    'referablePrecision', ...
    'referableF1', ...
    'ordinalAccuracy', ...
    'ordinalGrade4Recall', ...
    'grade4ProbThreshold', ...
    'thresholds', ...
    'results');

fprintf('\nSaved:\n');
fprintf('  %s\n', fullfile(modelDir, 'official_test_predictions.csv'));
fprintf('  %s\n', fullfile(modelDir, 'official_test_report.mat'));