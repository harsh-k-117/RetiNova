% EVALUATEDRDETECTONIDRID  Run the imported drdetect ONNX model across
% IDRiD B.Disease Grading and compare predictions to real CSV labels.
%
% Reports overall accuracy, a confusion matrix, and per-class
% precision/recall (grades 0-4).
%
% Same unconfirmed preprocessing assumption as testDrdetectONNX.m:
% standard ImageNet mean/std normalization. If accuracy looks
% suspiciously low/random, this is the first thing to question.

onnxPath = '/home/desktop/new s temp/RetiNova/data/efficientnet_b0_regression_512px.onnx';  % <-- set to your file
baseDir  = '/home/desktop/new s temp/RetiNova/data/B. Disease Grading/B. Disease Grading';
setType  = 'train';   % 'train' or 'test'

% --- Load labels ---
if strcmpi(setType, 'train')
    setFolder = 'a. Training Set';
    csvFile = fullfile(baseDir, '2. Groundtruths', ...
        'a. IDRiD_Disease Grading_Training Labels.csv');
else
    setFolder = 'b. Testing Set';
    csvFile = fullfile(baseDir, '2. Groundtruths', ...
        'b. IDRiD_Disease Grading_Testing Labels.csv');
end

labelsTable = readtable(csvFile);
varNames = labelsTable.Properties.VariableNames;

imgColIdx = find(contains(lower(varNames), 'image'), 1);
gradeColIdx = find(contains(lower(varNames), 'retinopathy') & contains(lower(varNames), 'grade'), 1);

if isempty(imgColIdx) || isempty(gradeColIdx)
    error(['Could not auto-detect image/grade columns. Actual columns are:\n%s'], ...
        strjoin(varNames, ', '));
end

imgNames = string(labelsTable.(varNames{imgColIdx}));
if ~endsWith(imgNames(1), '.jpg')
    imgNames = imgNames + ".jpg";
end
trueGrades = labelsTable.(varNames{gradeColIdx});

imgDir = fullfile(baseDir, '1. Original Images', setFolder);

% --- Import the network once ---
net = importNetworkFromONNX(onnxPath);

meanVals = reshape([0.485 0.456 0.406], 1, 1, 3);
stdVals  = reshape([0.229 0.224 0.225], 1, 1, 3);

numImages = numel(imgNames);
predGrades = zeros(numImages, 1);

for k = 1:numImages
    imgPath = fullfile(imgDir, imgNames(k));
    img = imread(imgPath);
    img = imresize(img, [512 512]);
    imgSingle = single(img) / 255;
    imgNorm = (imgSingle - meanVals) ./ stdVals;
    dlImg = dlarray(imgNorm, 'SSCB');

    if k == 1 && ~net.Initialized
        net = initialize(net, dlImg);
    end

    score = extractdata(predict(net, dlImg));
    predGrades(k) = round(min(max(score, 0), 4));

    if mod(k, 50) == 0
        fprintf('Processed %d / %d\n', k, numImages);
    end
end

% --- Metrics ---
accuracy = mean(predGrades == trueGrades);
fprintf('\nOverall accuracy: %.2f%% (%d/%d)\n', 100*accuracy, ...
    sum(predGrades == trueGrades), numImages);

classes = 0:4;
confMat = confusionmat(trueGrades, predGrades, 'Order', classes);
disp('Confusion matrix (rows = true, cols = predicted):');
disp(array2table(confMat, 'VariableNames', "Pred_" + string(classes), ...
    'RowNames', "True_" + string(classes)));

fprintf('\nPer-class precision / recall:\n');
for c = classes
    tp = confMat(c+1, c+1);
    fp = sum(confMat(:, c+1)) - tp;
    fn = sum(confMat(c+1, :)) - tp;

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);

    fprintf('Grade %d: precision = %.3f, recall = %.3f\n', c, precision, recall);
end