function cam = computeGradCAM(net, X, classIndex)
%COMPUTEGRADCAM Robust Grad-CAM for RetiNova 5-class DR grading.
%
%   CAM = COMPUTEGRADCAM(NET, X, CLASSINDEX)
%
%   Uses:
%       - a late spatial convolutional feature layer
%       - the pre-softmax classification score when available
%       - explicit FeatureLayer/ReductionLayer for MATLAB gradCAM
%       - custom Grad-CAM fallback
%       - robust percentile normalization
%
%   CAM is returned as HxW in [0,1].

cam = [];

if nargin < 3 || isempty(net) || isempty(X)
    return;
end

% -------------------------------------------------------------------------
% Normalize class index
% MATLAB uses 1..5 internally:
%   1 -> grade 0
%   2 -> grade 1
%   3 -> grade 2
%   4 -> grade 3
%   5 -> grade 4
% -------------------------------------------------------------------------

classIndex = round(double(classIndex));

if ~isfinite(classIndex)
    return;
end

classIndex = max(1, min(5, classIndex));

% -------------------------------------------------------------------------
% Prepare input
% -------------------------------------------------------------------------

try
    if isa(X, 'dlarray')
        Xnum = extractdata(X);

        if isa(Xnum, 'gpuArray')
            Xnum = gather(Xnum);
        end

        Xnum = single(Xnum);
    else
        Xnum = single(X);

        if isa(Xnum, 'gpuArray')
            Xnum = gather(Xnum);
        end
    end

    % Remove accidental singleton batch dimension.
    if ndims(Xnum) == 4 && size(Xnum,4) == 1
        Xnum = Xnum(:,:,:,1);
    end

    if ndims(Xnum) ~= 3
        return;
    end

catch
    return;
end

% -------------------------------------------------------------------------
% Find a good feature layer and classification/reduction layer.
% -------------------------------------------------------------------------

featureLayer = findBestFeatureLayer(net);
reductionLayer = findClassificationScoreLayer(net);

% -------------------------------------------------------------------------
% First attempt:
% MATLAB built-in Grad-CAM with explicitly selected layers.
%
% The current MATLAB gradCAM API supports FeatureLayer and ReductionLayer.
% -------------------------------------------------------------------------

if ~isempty(featureLayer)

    try

        if ~isempty(reductionLayer)

            cam = gradCAM( ...
                net, ...
                Xnum, ...
                classIndex, ...
                'FeatureLayer', featureLayer, ...
                'ReductionLayer', reductionLayer, ...
                'ExecutionEnvironment', 'auto');

        else

            cam = gradCAM( ...
                net, ...
                Xnum, ...
                classIndex, ...
                'FeatureLayer', featureLayer, ...
                'ExecutionEnvironment', 'auto');

        end

    catch
        cam = [];
    end
end

% -------------------------------------------------------------------------
% Custom fallback.
% -------------------------------------------------------------------------

if isempty(cam)

    try

        cam = customGradCAM( ...
            net, ...
            Xnum, ...
            classIndex, ...
            featureLayer, ...
            reductionLayer);

    catch
        cam = [];
    end
end

% -------------------------------------------------------------------------
% Final cleanup / normalization
% -------------------------------------------------------------------------

if isempty(cam)
    return;
end

try

    cam = double(cam);

    % Remove singleton dimensions.
    cam = squeeze(cam);

    if ndims(cam) ~= 2
        % Try extracting the first sample/channel if necessary.
        cam = cam(:,:,1);
    end

    % Positive Grad-CAM evidence only.
    cam = max(cam, 0);

    if isempty(cam) || ...
            all(~isfinite(cam(:))) || ...
            max(cam(:)) <= 0

        cam = [];
        return;
    end

    % Replace non-finite values.
    cam(~isfinite(cam)) = 0;

    % ---------------------------------------------------------------------
    % Resize to input resolution.
    % ---------------------------------------------------------------------

    targetSize = [size(Xnum,1), size(Xnum,2)];

    if ~isequal(size(cam), targetSize)

        cam = imresize( ...
            cam, ...
            targetSize, ...
            'bicubic');

    end

    % ---------------------------------------------------------------------
    % Robust normalization.
    %
    % Simple max normalization can make almost the entire CAM invisible
    % when one pixel is an extreme outlier.
    %
    % Use the 20th and 99th percentiles instead.
    % ---------------------------------------------------------------------

    lo = prctile(cam(:), 20);
    hi = prctile(cam(:), 99);

    if hi > lo

        cam = ...
            (cam - lo) ./ ...
            max(hi - lo, eps);

    else

        cam = mat2gray(cam);

    end

    cam = min(max(cam, 0), 1);

    % ---------------------------------------------------------------------
    % Mild contrast enhancement.
    %
    % Gamma < 1 makes moderate evidence more visible without creating
    % evidence where none exists.
    % ---------------------------------------------------------------------

    cam = cam .^ 0.75;

    % ---------------------------------------------------------------------
    % Mild spatial smoothing for a cleaner medical-image overlay.
    % ---------------------------------------------------------------------

    try

        cam = imgaussfilt(cam, 1.0);

    catch
        % imgaussfilt may not be available in some configurations.
    end

    % Re-normalize after smoothing.
    mx = max(cam(:));

    if mx > 0
        cam = cam ./ mx;
    end

    cam = min(max(cam, 0), 1);

catch
    cam = [];
end

end


% =========================================================================
% BUILT-IN / NETWORK FEATURE-LAYER DISCOVERY
% =========================================================================

function name = findBestFeatureLayer(net)

name = '';

if isempty(net)
    return;
end

try
    layers = net.Layers;
catch
    return;
end

% -------------------------------------------------------------------------
% Preferred:
% last ReLU with spatial output.
%
% MATLAB recommends a late ReLU or final convolutional feature layer for
% Grad-CAM feature extraction.
% -------------------------------------------------------------------------

for i = numel(layers):-1:1

    try

        if isa(layers(i), 'nnet.cnn.layer.ReLULayer')

            name = string(layers(i).Name);
            return;

        end

    catch
    end
end

% -------------------------------------------------------------------------
% Fallback:
% last convolution-like spatial layer.
% -------------------------------------------------------------------------

convTypes = { ...
    'nnet.cnn.layer.Convolution2DLayer', ...
    'nnet.cnn.layer.GroupedConvolution2DLayer', ...
    'nnet.cnn.layer.TransposedConvolution2DLayer'};

fallback = '';

for i = numel(layers):-1:1

    isConv = false;

    for t = 1:numel(convTypes)

        try

            if isa(layers(i), convTypes{t})

                isConv = true;
                break;

            end

        catch
        end

    end

    if ~isConv
        continue;
    end

    if isempty(fallback)
        fallback = string(layers(i).Name);
    end

    % Prefer a 3x3 or larger convolution.
    try

        if max(layers(i).FilterSize) >= 3

            name = string(layers(i).Name);
            return;

        end

    catch
    end

end

if ~isempty(fallback)
    name = fallback;
end

end


% =========================================================================
% CLASSIFICATION SCORE LAYER
% =========================================================================

function name = findClassificationScoreLayer(net)

name = '';

if isempty(net)
    return;
end

try
    layers = net.Layers;
catch
    layers = [];
end

if isempty(layers)
    return;
end

% -------------------------------------------------------------------------
% First preference:
% specifically named RetiNova grading head.
% -------------------------------------------------------------------------

for i = numel(layers):-1:1

    try

        layerName = string(layers(i).Name);

        if layerName == "grade_head"

            name = layerName;
            return;

        end

    catch
    end

end

% -------------------------------------------------------------------------
% Second preference:
% last fully connected layer.
%
% This should normally be the pre-softmax class score layer.
% -------------------------------------------------------------------------

for i = numel(layers):-1:1

    try

        if isa( ...
                layers(i), ...
                'nnet.cnn.layer.FullyConnectedLayer')

            name = string(layers(i).Name);
            return;

        end

    catch
    end

end

% -------------------------------------------------------------------------
% Final fallback:
% network output layer.
% -------------------------------------------------------------------------

try

    outputs = net.OutputNames;

    if ~isempty(outputs)

        name = string(outputs(1));

    end

catch
end

end


% =========================================================================
% CUSTOM GRAD-CAM
% =========================================================================

function cam = customGradCAM( ...
    net, ...
    X, ...
    classIndex, ...
    featureLayer, ...
    reductionLayer)

cam = [];

if isempty(featureLayer)
    return;
end

% -------------------------------------------------------------------------
% Make formatted single-image dlarray.
% -------------------------------------------------------------------------

dlX = dlarray( ...
    single(X), ...
    'SSC');

% -------------------------------------------------------------------------
% Compute gradients using dlfeval.
% -------------------------------------------------------------------------

if isempty(reductionLayer)

    camDl = dlfeval( ...
        @gradCamStepUsingOutput, ...
        net, ...
        dlX, ...
        featureLayer, ...
        classIndex);

else

    camDl = dlfeval( ...
        @gradCamStepUsingLayers, ...
        net, ...
        dlX, ...
        featureLayer, ...
        reductionLayer, ...
        classIndex);

end

% -------------------------------------------------------------------------
% Extract numeric CAM.
% -------------------------------------------------------------------------

cam = extractdata(camDl);

if isa(cam, 'gpuArray')
    cam = gather(cam);
end

cam = double(cam);

cam = squeeze(cam);

if ndims(cam) > 2
    cam = cam(:,:,1);
end

end


% =========================================================================
% CUSTOM GRADIENT STEP — EXPLICIT SCORE LAYER
% =========================================================================

function cam = gradCamStepUsingLayers( ...
    net, ...
    dlX, ...
    featureLayer, ...
    reductionLayer, ...
    classIndex)

% Forward the network and explicitly request:
%
%   1. pre-softmax classification score
%   2. spatial feature map
%
% Using forward() here keeps the intermediate activation inside the
% automatic-differentiation graph.

[scoreOutput, activations] = ...
    forward( ...
        net, ...
        dlX, ...
        'Outputs', ...
        [string(reductionLayer), string(featureLayer)]);

% Select the requested class from the pre-softmax score.
score = selectClassScore( ...
    scoreOutput, ...
    classIndex);

% Gradient of the class score with respect to the spatial feature map.
gradients = dlgradient( ...
    score, ...
    activations);

% Global-average-pool the gradients over spatial dimensions.
weights = mean( ...
    gradients, ...
    [1 2]);

% Weighted feature maps.
cam = sum( ...
    activations .* weights, ...
    3);

% Standard Grad-CAM ReLU.
cam = max(cam, 0);

% Remove batch dimension.
cam = cam(:,:,1,1);

end


% =========================================================================
% CUSTOM GRADIENT STEP — NETWORK OUTPUT FALLBACK
% =========================================================================

function cam = gradCamStepUsingOutput( ...
    net, ...
    dlX, ...
    featureLayer, ...
    classIndex)

[scoreOutput, activations] = ...
    forward( ...
        net, ...
        dlX, ...
        'Outputs', ...
        [string(net.OutputNames(1)), string(featureLayer)]);

% Network output may already be probabilities.
% We intentionally use the output directly here because this branch
% is only reached when a separate pre-softmax score layer could not
% be identified.

score = selectClassScore( ...
    scoreOutput, ...
    classIndex);

gradients = dlgradient( ...
    score, ...
    activations);

weights = mean( ...
    gradients, ...
    [1 2]);

cam = sum( ...
    activations .* weights, ...
    3);

cam = max(cam, 0);

cam = cam(:,:,1,1);

end


% =========================================================================
% SELECT CLASS SCORE
% =========================================================================

function score = selectClassScore(scores, classIndex)

sz = size(scores);

% Common dlnetwork classification output:
%   5 x 1
if numel(sz) >= 2 && sz(1) == 5

    score = scores(classIndex, 1);
    return;

end

% Another common arrangement:
%   1 x 5
if numel(sz) >= 2 && sz(2) == 5

    score = scores(1, classIndex);
    return;

end

% 1x1x5x1 or similar.
if numel(scores) >= 5

    flat = scores(:);

    score = flat(classIndex);
    return;

end

% Last-resort scalar.
score = scores(1);

end