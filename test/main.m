% TRAINIDRIDGRADER Train and cross-validate a five-grade MATLAB grader.
% Requires Deep Learning Toolbox, Image Processing Toolbox, Statistics and
% Machine Learning Toolbox, and a CUDA GPU. Prefers MATLAB EfficientNet-B0
% (ResNet-50 fallback) for GPU kernels; ONNX is last resort. The official
% 103-image test split is never read by this script.
%
% CHANGES vs previous version (not yet run end-to-end):
%   1. ONNX fallback now appends a softmax layer after the new 5-class head.
%      Without it, 'crossentropy' was applied to raw scores and the loss
%      went negative.
%   2. 3-fold CV instead of 5, MaxEpochs 10, ValidationPatience 3 (time budget).
%   3. scoreNetwork checks the whole layer list for a softmax instead of only
%      the last entry.
% Requires preprocessIDRiDImage.m (defined elsewhere in your project) on the path.

clear; clc;
rng(42, 'twister');

numFolds = 3;

if ~canUseGPU
    error(['No CUDA GPU visible to MATLAB. Install Parallel Computing Toolbox ', ...
        'and a supported NVIDIA driver, then run gpuDeviceTable.']);
end
g = gpuDevice;
memGB = double(g.AvailableMemory) / 2^30;
fprintf('GPU: %s | compute capability %s | %.2f GB free\n', ...
    string(g.Name), string(g.ComputeCapability), memGB);

projectDir = fileparts(mfilename('fullpath'));
dataRoot = fullfile(projectDir, 'data', 'B. Disease Grading', ...
    'B. Disease Grading');
archive = fullfile(projectDir, 'data', 'B. Disease Grading.zip');
trainImageDir = fullfile(dataRoot, '1. Original Images', 'a. Training Set');
trainCsv = fullfile(dataRoot, '2. Groundtruths', ...
    'a. IDRiD_Disease Grading_Training Labels.csv');
modelDir = fullfile(projectDir, 'models', 'idrid_grade5');
if ~isfolder(trainImageDir) || isempty(dir(fullfile(trainImageDir, '*.jpg')))
    if ~isfile(archive)
        error('IDRiD archive not found: %s', archive);
    end
    unzip(archive, fullfile(projectDir, 'data', 'B. Disease Grading'));
end
if ~isfile(trainCsv)
    error('Training labels not found: %s', trainCsv);
end
if ~isfolder(modelDir)
    mkdir(modelDir);
end

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
T = readtable(trainCsv);
imageNames = string(T.ImageName);
rawGrades = double(T.RetinopathyGrade);
n = numel(imageNames);
classNames = string(0:4);
labels = categorical(string(rawGrades), classNames);
if any(isundefined(labels)) || n ~= 413
    error('Expected 413 IDRiD training rows with grades in 0..4; found %d.', n);
end
filePaths = fullfile(trainImageDir, imageNames + ".jpg");
if any(~isfile(filePaths))
    missing = find(~isfile(filePaths), 1);
    error('Image not found: %s', filePaths(missing));
end

counts = countcats(labels);
fprintf('Training counts, grades 0..4: %s\n', mat2str(counts'));
inputSize = [512 512 3];
cachedPaths = cachePreprocessedImages(filePaths, imageNames, ...
    fullfile(modelDir, 'cache_512'), inputSize);
[~, backboneName] = buildGradingNetwork(inputSize, projectDir);
fprintf('Backbone: %s\n', backboneName);

% Square-root inverse-frequency weighting softens imbalance without allowing
% the 20 grade-1 examples to dominate every update.
classWeights = sqrt(median(counts) ./ max(counts, 1));
classWeights = classWeights / mean(classWeights);
classWeights = classWeights(:)';
fprintf('Class weights, grades 0..4: %s\n', mat2str(classWeights, 3));

augmenter = imageDataAugmenter( ...
    'RandRotation', [-12 12], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8]);
readCached = @(filename) readCachedNormalized(filename);
imds = imageDatastore(cachedPaths, 'Labels', labels, 'ReadFcn', readCached);
cv = cvpartition(labels, 'KFold', numFolds);
oofScores = nan(n, 5);
oofPredictions = nan(n, 1);
foldMetrics = table('Size', [numFolds 5], ...
    'VariableTypes', {'double','double','double','double','double'}, ...
    'VariableNames', {'Fold','Accuracy','QWK','MacroRecall','Grade4Recall'});
miniBatch = pickMiniBatchSize(g.AvailableMemory);

for fold = 1:numFolds
    fprintf('\nFold %d / %d\n', fold, numFolds);
    trainIdx = training(cv, fold);
    valIdx = test(cv, fold);
    foldTrainingRows = find(trainIdx);
    inner = cvpartition(labels(trainIdx), 'HoldOut', 0.15);
    innerTrainRows = foldTrainingRows(training(inner));
    earlyStopRows = foldTrainingRows(test(inner));
    balancedRows = balanceTrainingRows(innerTrainRows, rawGrades);
    trainDS = imageDatastore(cachedPaths(balancedRows), ...
        'Labels', labels(balancedRows), 'ReadFcn', readCached);
    trainDS.ReadSize = miniBatch;
    earlyStopDS = subset(imds, earlyStopRows);
    valDS = subset(imds, find(valIdx));
    trainAug = augmentedImageDatastore(inputSize(1:2), trainDS, ...
        'DataAugmentation', augmenter);
    trainAug.MiniBatchSize = miniBatch;
    earlyStopAug = augmentedImageDatastore(inputSize(1:2), earlyStopDS);
    earlyStopAug.MiniBatchSize = miniBatch;
    stepsPerEpoch = max(1, ceil(numel(balancedRows) / miniBatch));
    ckptDir = fullfile(modelDir, sprintf('checkpoints_fold-%d', fold));
    if ~isfolder(ckptDir)
        mkdir(ckptDir);
    end

    trained = false;
    batchTry = miniBatch;
    while ~trained
        net = buildGradingNetwork(inputSize, projectDir);
        options = makeGpuTrainOptions(batchTry, earlyStopAug, ...
            stepsPerEpoch, ckptDir);
        fprintf('Training on GPU, MiniBatchSize %d, ~%d steps/epoch\n', ...
            batchTry, max(1, ceil(numel(balancedRows) / batchTry)));
        try
            net = trainnet(trainAug, net, 'crossentropy', options);
            trained = true;
            miniBatch = batchTry;
        catch err
            if batchTry > 2 && isGpuMemoryError(err)
                nextBatch = max(2, batchTry / 2);
                fprintf('GPU out of memory at batch %d. Retrying with %d.\n', ...
                    batchTry, nextBatch);
                batchTry = nextBatch;
                trainAug.MiniBatchSize = batchTry;
                earlyStopAug.MiniBatchSize = batchTry;
                reset(gpuDevice);
            else
                rethrow(err);
            end
        end
    end
    scores = scoreNetwork(net, valDS, miniBatch);
    [~, scoreIndex] = max(scores, [], 2);
    predLabels = categorical(string(scoreIndex - 1), classNames);
    truth = rawGrades(valIdx);
    predicted = str2double(string(predLabels));
    oofScores(valIdx, :) = scores;
    oofPredictions(valIdx) = predicted;
    metrics = gradeMetrics(truth, predicted);
    foldMetrics{fold, :} = [fold, metrics.accuracy, metrics.qwk, ...
        metrics.macroRecall, metrics.grade4Recall];
    fprintf('accuracy %.3f | QWK %.3f | macro recall %.3f | grade-4 recall %.3f\n', ...
        metrics.accuracy, metrics.qwk, metrics.macroRecall, metrics.grade4Recall);

    foldModelPath = fullfile(modelDir, sprintf('fold-%d.mat', fold));
    save(foldModelPath, 'net', 'inputSize', 'classNames', 'backboneName', ...
        'fold', '-v7.3');
    fprintf('Saved %s\n', foldModelPath);
    clear net;
end

oofMetrics = gradeMetrics(rawGrades, oofPredictions);
expectedGrades = oofScores * (0:4)';
thresholds = fitGradeThresholds(expectedGrades, rawGrades);
ordinalPredictions = decodeGrades(expectedGrades, thresholds);
ordinalMetrics = gradeMetrics(rawGrades, ordinalPredictions);
confusion = confusionmat(rawGrades, ordinalPredictions, 'Order', 0:4);
disp('Out-of-fold confusion matrix after ordinal threshold fitting:');
disp(array2table(confusion, 'VariableNames', "Pred_" + classNames, ...
    'RowNames', "True_" + classNames));
fprintf('\nArgmax OOF: accuracy %.3f | QWK %.3f | macro recall %.3f | grade-4 recall %.3f\n', ...
    oofMetrics.accuracy, oofMetrics.qwk, oofMetrics.macroRecall, oofMetrics.grade4Recall);
fprintf('Ordinal OOF: accuracy %.3f | QWK %.3f | macro recall %.3f | grade-4 recall %.3f\n', ...
    ordinalMetrics.accuracy, ordinalMetrics.qwk, ...
    ordinalMetrics.macroRecall, ordinalMetrics.grade4Recall);
fprintf('Fitted ordinal thresholds: %s\n', mat2str(thresholds, 3));
writetable(foldMetrics, fullfile(modelDir, 'fold_metrics.csv'));
save(fullfile(modelDir, 'cross_validation.mat'), 'oofScores', ...
    'oofPredictions', 'ordinalPredictions', 'expectedGrades', 'thresholds', ...
    'rawGrades', 'confusion', 'oofMetrics', 'ordinalMetrics', ...
    'foldMetrics', 'inputSize', 'classNames', 'backboneName', 'classWeights');
fprintf('\nSaved %d fold models and OOF report under:\n%s\n', numFolds, modelDir);

function [net, backboneName] = buildGradingNetwork(inputSize, projectDir)
net = [];
backboneName = '';
if exist('imagePretrainedNetwork', 'file') == 2
    modelNames = ["efficientnetb0", "resnet50"];
    for i = 1:numel(modelNames)
        modelName = modelNames(i);
        try
            net = imagePretrainedNetwork(modelName, 'NumClasses', 5);
            net = replaceImageInput(net, inputSize);
            backboneName = char(modelName);
            return
        catch err
            fprintf('%s not used: %s\n', modelName, err.message);
            net = [];
        end
    end
end
onnxPath = fullfile(projectDir, 'data', 'efficientnet_b0_regression_512px.onnx');
if ~isfile(onnxPath)
    error(['Install the EfficientNet-B0 or ResNet-50 support package, or place ', ...
        'efficientnet_b0_regression_512px.onnx under data/.']);
end
baseNet = importNetworkFromONNX(onnxPath);
lgraph = layerGraph(baseNet);
head = fullyConnectedLayer(5, 'Name', 'grade_head', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10, ...
    'WeightsInitializer', 'he');
lgraph = replaceLayer(lgraph, 'x_backbone_classifie', head);
% Softmax so 'crossentropy' receives probabilities (fixes negative loss).
lgraph = addLayers(lgraph, softmaxLayer('Name', 'grade_softmax'));
lgraph = connectLayers(lgraph, 'grade_head', 'grade_softmax');
net = dlnetwork(lgraph);
backboneName = 'efficientnet_b0_onnx';
end

function net = replaceImageInput(net, inputSize)
lgraph = layerGraph(net);
layers = lgraph.Layers;
isInput = arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.ImageInputLayer'), layers);
if ~any(isInput)
    error('No image input layer found.');
end
oldLayer = layers(find(isInput, 1));
newLayer = imageInputLayer(inputSize, 'Name', oldLayer.Name, ...
    'Normalization', 'none');
lgraph = replaceLayer(lgraph, oldLayer.Name, newLayer);
net = dlnetwork(lgraph);
end

function options = makeGpuTrainOptions(miniBatch, valData, stepsPerEpoch, ckptDir)
common = {'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', miniBatch, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency', stepsPerEpoch, ...
    'ValidationPatience', 3, ...
    'ExecutionEnvironment', 'gpu', ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'Plots', 'none', ...
    'OutputNetwork', 'best-validation-loss', ...
    'CheckpointPath', ckptDir, ...
    'CheckpointFrequency', 1};
try
    options = trainingOptions('adam', common{:}, ...
        'CheckpointFrequencyUnit', 'epoch', 'DispatchInBackground', true);
    return
catch
end
try
    options = trainingOptions('adam', common{:}, 'DispatchInBackground', true);
    return
catch
end
options = trainingOptions('adam', common{:});
end

function miniBatch = pickMiniBatchSize(availableBytes)
gb = double(availableBytes) / 2^30;
if gb >= 8
    miniBatch = 16;
elseif gb >= 5
    miniBatch = 8;
elseif gb >= 3
    miniBatch = 4;
else
    miniBatch = 2;
end
end

function tf = isGpuMemoryError(err)
msg = lower(string(err.message) + " " + string(err.identifier));
tf = contains(msg, "out of memory") || contains(msg, "oom") || ...
    (contains(msg, "gpu") && contains(msg, "memory"));
end

function cachedPaths = cachePreprocessedImages(filePaths, imageNames, cacheDir, inputSize)
if ~isfolder(cacheDir)
    mkdir(cacheDir);
end
cachedPaths = strings(size(filePaths));
fprintf('Caching 512 px training images under %s\n', cacheDir);
for i = 1:numel(filePaths)
    outFile = fullfile(cacheDir, imageNames(i) + ".png");
    cachedPaths(i) = outFile;
    if isfile(outFile)
        continue
    end
    rgb = preprocessIDRiDImage(filePaths(i), inputSize, false);
    imwrite(rgb, outFile);
    if mod(i, 25) == 0 || i == numel(filePaths)
        fprintf('  cached %d / %d\n', i, numel(filePaths));
    end
end
end

function I = readCachedNormalized(filename)
pixels = im2single(imread(filename));
meanRGB = reshape(single([0.485 0.456 0.406]), 1, 1, 3);
stdRGB = reshape(single([0.229 0.224 0.225]), 1, 1, 3);
I = (pixels - meanRGB) ./ stdRGB;
end

function rows = balanceTrainingRows(sourceRows, grades)
% Sample every class to the largest class in this fold for each epoch.
foldGrades = grades(sourceRows);
targetCount = max(histcounts(foldGrades, -0.5:1:4.5));
rows = zeros(0, 1);
for grade = 0:4
    classRows = sourceRows(foldGrades == grade);
    rows = [rows; classRows(randsample(numel(classRows), targetCount, true))]; %#ok<AGROW>
end
rows = rows(randperm(numel(rows)));
end

function scores = scoreNetwork(net, ds, miniBatch)
numImages = numel(ds.Files);
scores = zeros(numImages, 5, 'single');
sample = single(readimage(ds, 1));
H = size(sample, 1);
W = size(sample, 2);
hasSoftmax = any(arrayfun(@(l) isa(l, 'nnet.cnn.layer.SoftmaxLayer'), net.Layers));
useSoftmax = ~hasSoftmax;
for startIdx = 1:miniBatch:numImages
    stopIdx = min(startIdx + miniBatch - 1, numImages);
    nb = stopIdx - startIdx + 1;
    batch = zeros(H, W, 3, nb, 'single');
    for j = 1:nb
        batch(:, :, :, j) = single(readimage(ds, startIdx + j - 1));
    end
    X = dlarray(gpuArray(batch), 'SSCB');
    out = predict(net, X);
    if useSoftmax
        out = softmax(out);
    end
    P = reshape(gather(extractdata(out)), 5, []);
    scores(startIdx:stopIdx, :) = P';
end
end

function m = gradeMetrics(truth, predicted)
valid = isfinite(predicted);
truth = truth(valid);
predicted = predicted(valid);
cm = confusionmat(truth, predicted, 'Order', 0:4);
recall = diag(cm) ./ max(sum(cm, 2), 1);
n = sum(cm(:));
[r, c] = ndgrid(0:4, 0:4);
w = (r - c).^2 / 16;
observed = sum(w(:) .* cm(:)) / max(n, 1);
expectedCounts = sum(cm, 2) * sum(cm, 1) / max(n, 1);
expected = sum(w(:) .* expectedCounts(:)) / max(n, 1);
m.accuracy = sum(diag(cm)) / max(n, 1);
m.qwk = 1 - observed / max(expected, eps);
m.macroRecall = mean(recall);
m.grade4Recall = recall(5);
end

function thresholds = fitGradeThresholds(expectedGrades, truth)
% Coordinate search on training-only out-of-fold predictions.
thresholds = 0.5:1:3.5;
best = gradeMetrics(truth, decodeGrades(expectedGrades, thresholds)).qwk;
for pass = 1:5
    changed = false;
    for j = 1:4
        lo = -0.25;
        hi = 4.25;
        if j > 1
            lo = thresholds(j-1) + 0.02;
        end
        if j < 4
            hi = thresholds(j+1) - 0.02;
        end
        candidates = linspace(lo, hi, 61);
        for candidate = candidates
            trial = thresholds;
            trial(j) = candidate;
            score = gradeMetrics(truth, decodeGrades(expectedGrades, trial)).qwk;
            if score > best
                best = score;
                thresholds = trial;
                changed = true;
            end
        end
    end
    if ~changed
        break
    end
end
end

function predicted = decodeGrades(expectedGrades, thresholds)
predicted = zeros(size(expectedGrades));
for threshold = thresholds
    predicted = predicted + (expectedGrades > threshold);
end
end