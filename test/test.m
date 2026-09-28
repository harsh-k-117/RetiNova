imgFolder='/home/desktop/new s temp/RetiNova/data/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/';
files = dir(fullfile(imgFolder, '*.jpg'));
lap = fspecial('laplacian');


im = imread(fullfile(imgFolder, 'IDRiD_017.jpg'));
imshow(im);   % click to find a vessel-rich spot for this specific image, e.g. near the optic disc
[x, y] = ginput(1);   % click on a vessel-dense spot after imshow(im) above
crop = imcrop(im, [x-150, y-150, 300, 300]);
figure; imshow(crop); title('IDRiD_017.jpg - vessel region');