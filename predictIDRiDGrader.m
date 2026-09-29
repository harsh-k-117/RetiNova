function result = predictIDRiDGrader(imageInput, modelDir)
%PREDICTIDRIDGRADER Average trained fold probabilities, or run demo mode.
%   Uses however many fold-*.mat files are present (5-fold after the
%   current trainer). Softmax is applied only if the net has no softmax layer.
%   The final grade uses the frozen decoder saved by the trainer: ordinal
%   expected-grade thresholds, then the grade-4 override
%   (argmax == grade 4 OR P(grade 4) >= grade4ProbThreshold).

if nargin < 2 || isempty(modelDir)
    projectDir = fileparts(mfilename('fullpath'));
    modelDir = fullfile(projectDir, 'models', 'idrid_grade5');
end
projectDir = fileparts(mfilename('fullpath'));
if ischar(imageInput) || isstring(imageInput)
    image = imread(imageInput);
else
    image = imageInput;
end
if ndims(image) == 2
    image = repmat(image, 1, 1, 3);
elseif size(image, 3) == 4
    image = image(:, :, 1:3);
end
quality = assessQuality(image);
gradeNames = ["No apparent DR", "Mild NPDR", "Moderate NPDR", ...
    "Severe NPDR", "Proliferative DR"];
emptyResult = struct('isUsable', false, 'grade', [], 'confidence', [], ...
    'probabilities', [], 'expectedGrade', [], 'cam', [], ...
    'overlay', [], 'demoMode', false, 'gradeName', '', ...
    'referable', false, 'quality', quality, 'message', '', ...
    'numFolds', 0, 'backboneName', '', 'oofAccuracy', [], 'oofQwk', []);

if ~quality.isUsable
    emptyResult.message = 'Image quality is insufficient. Recapture the fundus image.';
    result = emptyResult;
    return
end

foldPaths = discoverFoldModels(modelDir);
calibrationPath = fullfile(modelDir, 'cross_validation.mat');
if ~isempty(foldPaths) && isfile(calibrationPath)
    result = predictTrainedEnsemble(image, modelDir, foldPaths, ...
        quality, gradeNames);
else
    result = predictDemo(image, projectDir, modelDir, quality, gradeNames);
end
end

function result = predictTrainedEnsemble(image, modelDir, foldPaths, ...
    quality, gradeNames)
persistent cache
nFolds = numel(foldPaths);
stamp = join(string(foldPaths), '|');
if isempty(cache) || cache.stamp ~= stamp
    cache = struct('stamp', stamp, 'nets', {{}}, 'inputSize', [], ...
        'thresholds', [], 'grade4ProbThreshold', [], ...
        'backboneName', '', 'oofAccuracy', [], 'oofQwk', []);
    cal = load(fullfile(modelDir, 'cross_validation.mat'));
    cache.thresholds = cal.thresholds;
    % Older models saved before the hybrid decoder have no such field;
    % they keep the ordinal-only behaviour.
    if isfield(cal, 'grade4ProbThreshold')
        cache.grade4ProbThreshold = cal.grade4ProbThreshold;
    end
    if isfield(cal, 'ordinalMetrics')
        cache.oofAccuracy = cal.ordinalMetrics.accuracy;
        cache.oofQwk = cal.ordinalMetrics.qwk;
    end
    if isfield(cal, 'backboneName')
        cache.backboneName = char(string(cal.backboneName));
    end
    for i = 1:nFolds
        S = load(foldPaths{i}, 'net', 'inputSize', 'backboneName');
        cache.nets{i} = S.net;
        if isempty(cache.inputSize)
            cache.inputSize = S.inputSize;
        end
        if isempty(cache.backboneName) && isfield(S, 'backboneName')
            cache.backboneName = char(string(S.backboneName));
        end
    end
end

% Preprocessing does not depend on the fold, so do it once.
X = preprocessIDRiDImage(image, cache.inputSize);
meanScores = zeros(1, 5);
for i = 1:nFolds
    meanScores = meanScores + classScores(cache.nets{i}, X) / nFolds;
end
probabilities = meanScores / max(sum(meanScores), eps);
expectedGrade = probabilities * (0:4)';
grade = sum(expectedGrade > cache.thresholds);

% Grade-4 override, identical to fitGrade4ProbThreshold in the trainer.
[~, topClass] = max(probabilities);
grade4Override = false;
if ~isempty(cache.grade4ProbThreshold) && ...
        (topClass == 5 || probabilities(5) >= cache.grade4ProbThreshold)
    grade4Override = grade < 4;
    grade = 4;
end

cam = networkOrLesionCam(cache.nets{1}, image, cache.inputSize, grade, probabilities);
message = sprintf('Trained %d-fold ensemble. Expected grade %.2f.', nFolds, expectedGrade);
if grade4Override
    message = [message ' Grade-4 probability override applied.'];
end
result = packResult(false, grade, probabilities, expectedGrade, cam, image, ...
    quality, gradeNames, message, nFolds, cache.backboneName, ...
    cache.oofAccuracy, cache.oofQwk);
end

function result = predictDemo(image, projectDir, modelDir, quality, gradeNames)
persistent demoNet demoInputSize
if exist('demoGradeScores', 'file') ~= 2
    error('predictIDRiDGrader:noDemoHelper', ...
        ['Trained models were not found (need fold-*.mat and ' ...
        'cross_validation.mat in %s) and the demo helper ' ...
        'demoGradeScores.m is not on the MATLAB path. Run ' ...
        'trainIDRiDGrader.m or add demoGradeScores.m to the path.'], ...
        modelDir);
end
if isempty(demoNet)
    [demoNet, demoInputSize] = loadDemoGradingNetwork(projectDir);
end
[probabilities, expectedGrade] = demoGradeScores(image);
thresholds = 0.5:1:3.5;
grade = sum(expectedGrade > thresholds);
cam = networkOrLesionCam(demoNet, image, demoInputSize, grade, probabilities);
message = 'Demo mode: trained fold models were not found.';
result = packResult(true, grade, probabilities, expectedGrade, cam, image, ...
    quality, gradeNames, message, 0, '', [], []);
end

function scores = classScores(net, X)
out = predict(net, dlarray(single(X), 'SSCB'));
if ~networkHasSoftmax(net)
    out = softmax(out);
end
scores = double(extractdata(out));
scores = scores(:)';
if numel(scores) < 5
    scores(end+1:5) = 0;
else
    scores = scores(1:5);
end
total = sum(scores);
if total > 0
    scores = scores / total;
end
end

function tf = networkHasSoftmax(net)
tf = false;
try
    tf = any(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.SoftmaxLayer'), net.Layers));
catch
end
end

function paths = discoverFoldModels(modelDir)
paths = {};
for fold = 1:10
    modelPath = fullfile(modelDir, sprintf('fold-%d.mat', fold));
    if ~isfile(modelPath)
        break
    end
    paths{end+1} = modelPath; %#ok<AGROW>
end
end

function cam = networkOrLesionCam(net, image, inputSize, grade, probabilities)
cam = [];
if ~isempty(net) && ~isempty(inputSize)
    try
        X = preprocessIDRiDImage(image, inputSize);
        camClass = evidenceClassIndex(grade, probabilities);
        cam = computeGradCAM(net, X, camClass);
        if ~isempty(cam)
            cam = imresize(cam, [size(image, 1), size(image, 2)]);
        end
    catch
        cam = [];
    end
end
lesion = lesionSaliencyMap(image);
if isempty(cam)
    cam = lesion;
else
    cam = max(focusEvidence(cam), 0.85 * focusEvidence(lesion));
end
cam = suppressOpticDisc(image, cam);
cam = cam / max(max(cam(:)), eps);
end

function camClass = evidenceClassIndex(grade, probabilities)
% Grad-CAM of grade 0 lights up the disc/field, not lesions. Prefer a
% disease class when the model is not clearly on grade 0.
if nargin < 2 || isempty(probabilities)
    camClass = max(2, min(5, grade + 1));
    return
end
p = probabilities(:)';
if numel(p) < 5
    p(end+1:5) = 0;
end
[~, argmaxClass] = max(p);
if argmaxClass >= 2
    camClass = argmaxClass;
elseif p(2) >= p(1)
    camClass = 2;
else
    camClass = max(2, min(5, grade + 1));
end
end

function cam = focusEvidence(cam)
cam = double(cam);
cam = max(cam, 0);
peak = max(cam(:));
if peak <= 0
    return
end
cam = cam / peak;
vals = cam(cam > 0);
if numel(vals) >= 32
    cam = max(0, cam - prctile(vals, 65));
    peak = max(cam(:));
    if peak > 0
        cam = cam / peak;
    end
end
cam = cam .^ 1.4;
end

function cam = suppressOpticDisc(image, cam)
if isempty(cam)
    return
end
rgb = im2double(image(:, :, 1:min(3, size(image, 3))));
gray = rgb2gray(rgb);
cam = imresize(double(cam), size(gray));
bright = gray > prctile(gray(:), 99.3);
bright = bwareaopen(bright, max(16, round(numel(gray) * 0.0004)));
if ~any(bright(:))
    return
end
disc = imdilate(bright, strel('disk', max(6, round(min(size(gray)) * 0.028))));
weight = 1 - 0.82 * imgaussfilt(double(disc), max(3, min(size(gray)) / 70));
cam = cam .* weight;
end

function result = packResult(demoMode, grade, probabilities, expectedGrade, ...
    cam, image, quality, gradeNames, message, numFolds, backboneName, ...
    oofAccuracy, oofQwk)
grade = min(4, max(0, round(double(grade))));
confidence = probabilities(grade + 1);
overlay = blendCamOverlay(image, cam, 0.62);
result = struct('isUsable', true, 'grade', grade, ...
    'confidence', double(confidence), 'probabilities', double(probabilities), ...
    'expectedGrade', double(expectedGrade), 'cam', cam, 'overlay', overlay, ...
    'demoMode', demoMode, 'gradeName', char(gradeNames(grade + 1)), ...
    'referable', grade >= 2, 'quality', quality, 'message', message, ...
    'numFolds', numFolds, 'backboneName', char(string(backboneName)), ...
    'oofAccuracy', oofAccuracy, 'oofQwk', oofQwk);
end