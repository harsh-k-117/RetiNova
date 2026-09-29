function I = preprocessIDRiDImage(input, inputSize, normalize)
%PREPROCESSIDRIDIMAGE Shared fundus preprocessing for training and inference.
%   I = preprocessIDRiDImage(filename, inputSize) returns an HxWx3 single
%   image normalized with ImageNet statistics.
%   I = preprocessIDRiDImage(..., false) returns uint8 RGB after crop,
%   illumination correction, and resize, for training caches.

if nargin < 3
    normalize = true;
end

if ischar(input) || isstring(input)
    input = imread(input);
end
if ndims(input) == 2
    input = repmat(input, 1, 1, 3);
elseif size(input, 3) == 4
    input = input(:, :, 1:3);
end
if size(input, 3) ~= 3
    error('Expected a grayscale or RGB fundus image.');
end
if ~isa(input, 'uint8')
    input = im2uint8(input);
end

% Remove the black surround while retaining the complete retinal field.
gray = rgb2gray(input);
mask = imfill(imclose(gray > 8, strel('disk', 12, 0)), 'holes');
components = bwconncomp(mask);
if components.NumObjects > 0
    sizes = cellfun(@numel, components.PixelIdxList);
    [~, largest] = max(sizes);
    component = false(size(mask));
    component(components.PixelIdxList{largest}) = true;
    stats = regionprops(component, 'BoundingBox');
    box = stats(1).BoundingBox;
    x1 = max(1, floor(box(1)));
    y1 = max(1, floor(box(2)));
    x2 = min(size(input, 2), ceil(box(1) + box(3)));
    y2 = min(size(input, 1), ceil(box(2) + box(4)));
    input = input(y1:y2, x1:x2, :);
end

% Square-pad the retinal crop so resizing does not distort its aspect ratio.
h = size(input, 1);
w = size(input, 2);
side = max(h, w);
padY = side - h;
padX = side - w;
input = padarray(input, [floor(padY/2) floor(padX/2)], 0, 'pre');
input = padarray(input, [ceil(padY/2) ceil(padX/2)], 0, 'post');
input = imresize(input, inputSize(1:2));

% Mild per-channel illumination correction, identical at train and inference.
pixels = single(input) / 255;
for channel = 1:3
    plane = pixels(:, :, channel);
    background = imgaussfilt(plane, max(3, inputSize(1) / 16));
    plane = plane - background + 0.5;
    lo = prctile(plane(:), 1);
    hi = prctile(plane(:), 99);
    if hi > lo
        plane = (plane - lo) / (hi - lo);
    end
    pixels(:, :, channel) = min(max(plane, 0), 1);
end
if ~normalize
    I = im2uint8(pixels);
    return
end
meanRGB = reshape(single([0.485 0.456 0.406]), 1, 1, 3);
stdRGB = reshape(single([0.229 0.224 0.225]), 1, 1, 3);
I = (pixels - meanRGB) ./ stdRGB;
end
