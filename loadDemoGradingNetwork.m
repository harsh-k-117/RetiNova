function [net, inputSize] = loadDemoGradingNetwork(projectDir)
%LOADDEMOGRADINGNETWORK Untrained 5-class head on a pretrained backbone.
%   Used only when fold models are missing so the UI and Grad-CAM still run.

if nargin < 1 || isempty(projectDir)
    projectDir = fileparts(mfilename('fullpath'));
end
inputSize = [512 512 3];
net = [];
if exist('imagePretrainedNetwork', 'file') == 2
    modelNames = ["efficientnetb0", "resnet50"];
    for i = 1:numel(modelNames)
        try
            net = imagePretrainedNetwork(modelNames(i), 'NumClasses', 5);
            net = replaceImageInput(net, inputSize);
            return
        catch
            net = [];
        end
    end
end
onnxPath = fullfile(projectDir, 'data', 'efficientnet_b0_regression_512px.onnx');
if isfile(onnxPath) && exist('importNetworkFromONNX', 'file') == 2
    try
        baseNet = importNetworkFromONNX(onnxPath);
        lgraph = layerGraph(baseNet);
        head = fullyConnectedLayer(5, 'Name', 'grade_head');
        lgraph = replaceLayer(lgraph, 'x_backbone_classifie', head);
        net = dlnetwork(lgraph);
        return
    catch
        net = [];
    end
end
end

function net = replaceImageInput(net, inputSize)
lgraph = layerGraph(net);
layers = lgraph.Layers;
isInput = arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.ImageInputLayer'), layers);
if ~any(isInput)
    return
end
oldLayer = layers(find(isInput, 1));
newLayer = imageInputLayer(inputSize, 'Name', oldLayer.Name, ...
    'Normalization', 'none');
lgraph = replaceLayer(lgraph, oldLayer.Name, newLayer);
net = dlnetwork(lgraph);
end
