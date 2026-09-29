imgFolder='/home/desktop/new s temp/RetiNova/data/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set/';
files = dir(fullfile(imgFolder, '*.jpg'));

allQ = struct('sharpness', {}, 'brightness', {}, 'contrast', {}, 'over', {}, 'under', {});
for i = 1:length(files)
    im = imread(fullfile(imgFolder, files(i).name));
    q = assessQuality(im);
    allQ(i).sharpness  = q.sharpness;
    allQ(i).brightness = q.brightness;
    allQ(i).contrast   = q.contrast;
    allQ(i).over       = q.overexposedFrac;
    allQ(i).under      = q.underexposedFrac;
end

figure;
subplot(2,2,1); histogram([allQ.sharpness]);  title('Sharpness distribution');
subplot(2,2,2); histogram([allQ.brightness]); title('Brightness distribution');
subplot(2,2,3); histogram([allQ.contrast]);   title('Contrast distribution');
subplot(2,2,4); histogram([allQ.over]+[allQ.under]); title('Exposure fraction distribution');


sharpVals = [allQ.sharpness];
brightVals = [allQ.brightness];
contrastVals = [allQ.contrast];
exposureVals = [allQ.over] + [allQ.under];

fprintf('Sharpness percentiles:  %s\n', mat2str(round(prctile(sharpVals, [1 5 25 50 75 95 99]),1)));
fprintf('Brightness percentiles: %s\n', mat2str(round(prctile(brightVals, [1 5 25 50 75 95 99]),1)));
fprintf('Contrast percentiles:   %s\n', mat2str(round(prctile(contrastVals, [1 5 25 50 75 95 99]),1)));
fprintf('Exposure percentiles:   %s\n', mat2str(prctile(exposureVals, [1 5 25 50 75 95 99])));

usableFlags = arrayfun(@(q) (q.sharpness > 7) && (q.brightness > 40 && q.brightness < 140) ...
    && (q.contrast > 10) && ((q.over + q.under) < 0.01), allQ);
fprintf('%d / %d images pass with new thresholds\n', sum(usableFlags), numel(usableFlags));
failIdx = find(~usableFlags);

