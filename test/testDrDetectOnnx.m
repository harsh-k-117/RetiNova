% TESTDRDETECTONNX  Import the adarshcod30/drdetect-dr-screening ONNX
% checkpoint into MATLAB and run inference on a single test image.
%
% Source: https://huggingface.co/adarshcod30/drdetect-dr-screening
% File needed: efficientnet_b0_regression_512px.onnx (download manually
% from the "Files and versions" tab and set onnxPath below)
%
% LICENSE: research-use-only per the model card. Fine for testing your
% import/inference pipeline; not licensed for clinical/commercial use.
%
% UNCONFIRMED ASSUMPTIONS (flagged, not verified against the source repo):
%   1. Preprocessing: using standard ImageNet mean/std normalization,
%      since the model card does not document its own preprocessing
%      constants (it defers to an internal run_pipeline() function).
%   2. Decoding: rounding the regression output to the nearest integer
%      grade in [0,4]. The model card does not state the exact decoding
%      rule used in their own pipeline.
% Check github.com/adarshcod30/Diabetic-Retinopathy-Detection (src/
% drdetect/serve/pipeline.py) for the real values before trusting output.

onnxPath = '/home/desktop/new s temp/RetiNova/data/efficientnet_b0_regression_512px.onnx';  % <-- set to your downloaded file
testImagePath = '/home/desktop/new s temp/RetiNova/data/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_055.jpg';% <-- set to a real test image

% --- Import the network ---
net = importNetworkFromONNX(onnxPath);

% --- Preprocess the image ---
img = imread(testImagePath);
img = imresize(img, [512 512]);
imgSingle = single(img) / 255;

% ASSUMPTION: standard ImageNet normalization (see caveat above)
meanVals = reshape([0.485 0.456 0.406], 1, 1, 3);
stdVals  = reshape([0.229 0.224 0.225], 1, 1, 3);
imgNorm = (imgSingle - meanVals) ./ stdVals;

% Format as dlarray: spatial-spatial-channel-batch
dlImg = dlarray(imgNorm, 'SSCB');

% importNetworkFromONNX may return an uninitialized network if the ONNX
% graph's input format/size was ambiguous - initialize it with real
% example data before predicting.
if ~net.Initialized
    net = initialize(net, dlImg);
end

% --- Run inference ---
score = predict(net, dlImg);
score = extractdata(score);

fprintf('Raw regression score: %.4f\n', score);

% ASSUMPTION: decoding rule (see caveat above)
predictedGrade = round(min(max(score, 0), 4));
fprintf('Decoded DR grade (0-4): %d\n', predictedGrade);