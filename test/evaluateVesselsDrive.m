% EVALUATEVESSELSDRIVE  Batch-run segmentVessels over the DRIVE test set.
%
% IMPORTANT: Run this script from the project root directory, or add the
% segmentation/ folder to your MATLAB path before running.
%
% ASSUMES the standard DRIVE distribution folder layout:
%   <drivePath>/test/images/01_test.tif ... 20_test.tif
%   <drivePath>/test/mask/01_test_mask.gif ... 20_test_mask.gif
%   <drivePath>/test/1st_manual/01_manual1.gif ... 20_manual1.gif
%
% If you're using a Kaggle repackaging or different naming, update the
% filename patterns below before running.

drivePath = '/home/desktop/new s temp/RetiNova/data/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set/';   % <-- set to your DRIVE root folder

imgDir  = fullfile(drivePath, 'test', 'images');
maskDir = fullfile(drivePath, 'test', 'mask');
gtDir   = fullfile(drivePath, 'test', '1st_manual');

ids = 1:20;
diceAll = zeros(numel(ids),1);
iouAll  = zeros(numel(ids),1);

for k = 1:numel(ids)
    id = ids(k);

    imgFile  = fullfile(imgDir,  sprintf('%02d_test.tif', id));
    maskFile = fullfile(maskDir, sprintf('%02d_test_mask.gif', id));
    gtFile   = fullfile(gtDir,   sprintf('%02d_manual1.gif', id));

    img     = imread(imgFile);
    fovMask = logical(imread(maskFile));
    gtMask  = logical(imread(gtFile));

    [~, ~, metrics] = segmentVessels(img, fovMask, gtMask);

    diceAll(k) = metrics.dice;
    iouAll(k)  = metrics.iou;

    fprintf('Image %02d: Dice = %.3f, IoU = %.3f\n', id, metrics.dice, metrics.iou);
end

fprintf('\nMean Dice = %.3f (+/- %.3f)\n', mean(diceAll), std(diceAll));
fprintf('Mean IoU  = %.3f (+/- %.3f)\n', mean(iouAll), std(iouAll));