function overlay = blendCamOverlay(image, cam, alpha)
%BLENDCAMOVERLAY Mix a focused heatmap onto an RGB fundus image.
%   Low activations are suppressed so the overlay marks peaks instead of
%   washing the whole retina with a jet colormap.

if nargin < 3 || isempty(alpha)
    alpha = 0.62;
end
alpha = min(max(double(alpha), 0), 1);
if ndims(image) == 2
    image = repmat(image, 1, 1, 3);
end
img = im2double(image(:, :, 1:3));
if isempty(cam)
    overlay = img;
    return
end
cam = imresize(double(cam), [size(img, 1), size(img, 2)]);
cam = max(cam, 0);
gray = rgb2gray(img);
fov = imgaussfilt(double(gray > 0.04), 4) > 0.25;
cam = cam .* fov;
vals = cam(fov & cam > 0);
if numel(vals) >= 32
    floorVal = prctile(vals, 72);
    cam = max(0, cam - floorVal);
end
peak = max(cam(:));
if peak > 0
    cam = (cam / peak) .^ 1.65;
end
cam(cam < 0.18) = 0;
cmap = hot(256);
idx = min(256, max(1, round(1 + cam * 255)));
heat = ind2rgb(idx, cmap);
weight = alpha * cam;
overlay = img .* (1 - weight) + heat .* weight;
overlay = min(max(overlay, 0), 1);
end
