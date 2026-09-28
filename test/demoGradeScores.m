function [probabilities, expectedGrade] = demoGradeScores(image)
%DEMOGRADESCORES Placeholder 5-class scores from simple fundus cues.
%   Not a trained model. Used so the UI has a grade while weights are missing.

rgb = im2double(imresize(image, [384 384]));
[h, w, ~] = size(rgb);
[x, y] = meshgrid(1:w, 1:h);
mask = (x - (w + 1) / 2).^2 + (y - (h + 1) / 2).^2 < (0.47 * min(h, w))^2;
hsv = rgb2hsv(rgb);
v = hsv(:, :, 3);
s = hsv(:, :, 2);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);
exudate = mask & v > 0.72 & s < 0.55 & r >= g & g > b;
hemorrhage = mask & r > g * 1.12 & b < 0.40 & v > 0.08 & v < 0.48;
exFrac = mean(exudate(mask));
hemFrac = mean(hemorrhage(mask));
severity = min(4, max(0, 90 * exFrac + 140 * hemFrac));
centers = 0:4;
logits = -((severity - centers) .^ 2) / 0.85;
logits = logits - max(logits);
probabilities = exp(logits);
probabilities = probabilities / sum(probabilities);
expectedGrade = probabilities * (0:4)';
end
