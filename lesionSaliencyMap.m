function cam = lesionSaliencyMap(image)
%LESIONSALIENCYMAP Heuristic fundus heatmap when network Grad-CAM is unavailable.
%   Highlights bright exudate-like pixels and dark red hemorrhage-like pixels.

if ndims(image) == 2
    image = repmat(image, 1, 1, 3);
end
rgb = im2double(image(:, :, 1:3));
[h, w, ~] = size(rgb);
[x, y] = meshgrid(1:w, 1:h);
cx = (w + 1) / 2;
cy = (h + 1) / 2;
radius = 0.47 * min(h, w);
mask = (x - cx).^2 + (y - cy).^2 < radius^2;
hsv = rgb2hsv(rgb);
v = hsv(:, :, 3);
s = hsv(:, :, 2);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);

disc = mask & v > prctile(v(mask), 99.2) & s < 0.45;
disc = imdilate(disc, strel('disk', max(8, round(min(h, w) * 0.03))));
exudate = mask & ~disc & v > 0.72 & s < 0.55 & r > 0.55 & r >= g & g > b;
hemorrhage = mask & ~disc & r > g * 1.12 & b < 0.40 & v > 0.08 & v < 0.48;
greenInv = mask .* max(0, mean(g(mask)) - g);
cam = 0.65 * imgaussfilt(double(exudate), max(2, min(h, w) / 80)) + ...
    0.90 * imgaussfilt(double(hemorrhage), max(2, min(h, w) / 90)) + ...
    0.15 * imgaussfilt(greenInv, max(3, min(h, w) / 40));
cam = cam .* double(mask);
peak = max(cam(:));
if peak > 0
    cam = cam / peak;
else
    cam = zeros(h, w);
end
end
