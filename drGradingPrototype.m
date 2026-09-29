function drGradingPrototype
%DRGRADINGPROTOTYPE MATLAB UI for DR grade review and Grad-CAM overlay.
%   Uses whatever fold-*.mat files exist under models/idrid_grade5.

projectDir = fileparts(mfilename('fullpath'));
modelDir = fullfile(projectDir, 'models', 'idrid_grade5');
state = struct('image', [], 'result', []);

fig = uifigure('Name', 'RetiNova | DR grade review', ...
    'Position', [50 40 1320 820], 'Color', [0.94 0.96 0.97]);
root = uigridlayout(fig, [3 1]);
root.RowHeight = {72, 36, '1x'};
root.Padding = [18 16 18 16];
root.RowSpacing = 10;

header = uipanel(root, 'BorderType', 'none', 'BackgroundColor', fig.Color);
hg = uigridlayout(header, [1 4]);
hg.ColumnWidth = {'1x', 150, 150, 170};
hg.Padding = [0 0 0 0];
hg.ColumnSpacing = 10;
uilabel(hg, 'Text', 'RetiNova  /  DR screening review', ...
    'FontSize', 22, 'FontWeight', 'bold', 'FontColor', [0.05 0.15 0.20]);
uibutton(hg, 'Text', 'Open fundus image', 'ButtonPushedFcn', @onChooseImage);
uibutton(hg, 'Text', 'Load sample image', 'ButtonPushedFcn', @onSampleImage);
uibutton(hg, 'Text', 'Analyze', 'ButtonPushedFcn', @onAnalyze);

banner = uilabel(root, 'Text', modelStatusText(modelDir), ...
    'FontSize', 12, 'FontColor', [0.15 0.25 0.30], 'WordWrap', 'on');

body = uigridlayout(root, [1 3]);
body.ColumnWidth = {'5x', '5x', '4.2x'};
body.ColumnSpacing = 14;

left = uipanel(body, 'Title', 'Fundus image', 'FontWeight', 'bold', ...
    'BackgroundColor', [1 1 1]);
leftGrid = uigridlayout(left, [1 1]);
axImage = uiaxes(leftGrid);
styleAxes(axImage);
title(axImage, 'Original');

mid = uipanel(body, 'Title', 'Grad-CAM evidence', 'FontWeight', 'bold', ...
    'BackgroundColor', [1 1 1]);
midGrid = uigridlayout(mid, [2 1]);
midGrid.RowHeight = {'1x', 48};
axCam = uiaxes(midGrid);
styleAxes(axCam);
title(axCam, 'Heatmap overlay');
sliderPanel = uigridlayout(midGrid, [1 2]);
sliderPanel.ColumnWidth = {90, '1x'};
sliderPanel.Padding = [8 4 8 4];
uilabel(sliderPanel, 'Text', 'Overlay');
alphaSlider = uislider(sliderPanel, 'Limits', [0 1], 'Value', 0.62, ...
    'MajorTicks', [0 0.5 1], 'ValueChangingFcn', @onAlphaChanging, ...
    'ValueChangedFcn', @onAlphaChanged);

right = uipanel(body, 'Title', 'Model output', 'FontWeight', 'bold', ...
    'BackgroundColor', [1 1 1]);
rightGrid = uigridlayout(right, [8 1]);
rightGrid.RowHeight = {58, 26, 26, 36, 8, '1x', 80, 36};
rightGrid.Padding = [14 12 14 12];
gradeLabel = uilabel(rightGrid, 'Text', 'Waiting for image', ...
    'FontSize', 20, 'FontWeight', 'bold', 'FontColor', [0.05 0.12 0.18], ...
    'WordWrap', 'on');
referLabel = uilabel(rightGrid, 'Text', 'Referral: —', ...
    'FontSize', 14, 'FontColor', [0.15 0.25 0.30]);
expectedLabel = uilabel(rightGrid, 'Text', 'Expected grade: —', ...
    'FontSize', 13, 'FontColor', [0.15 0.25 0.30], 'WordWrap', 'on');
qualityLabel = uilabel(rightGrid, 'Text', 'Image quality: not assessed', ...
    'FontSize', 13, 'FontColor', [0.15 0.25 0.30], 'WordWrap', 'on');
metricsLabel = uilabel(rightGrid, 'Text', '', ...
    'FontSize', 12, 'FontColor', [0.20 0.30 0.35], 'WordWrap', 'on');
axScores = uiaxes(rightGrid);
axScores.Box = 'off';
axScores.XTick = 1:5;
axScores.XTickLabel = {'0','1','2','3','4'};
axScores.XLim = [0.5 5.5];
axScores.YLim = [0 1];
axScores.YLabel.String = 'Score';
title(axScores, 'Grade probabilities');
if isprop(axScores, 'Toolbar')
    axScores.Toolbar.Visible = 'off';
end
try
    disableDefaultInteractivity(axScores);
catch
end
noteLabel = uilabel(rightGrid, 'Text', '', 'FontSize', 12, ...
    'FontColor', [0.20 0.30 0.35], 'WordWrap', 'on');
uilabel(rightGrid, 'Text', ...
    'Screening support only. A clinician must confirm the result.', ...
    'FontSize', 11, 'FontColor', [0.60 0.35 0.15], 'WordWrap', 'on');

    function onChooseImage(~, ~)
        [file, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff', ...
            'Fundus images (*.jpg, *.jpeg, *.png, *.tif)'}, ...
            'Select a fundus image');
        if isequal(file, 0)
            return
        end
        showAndAnalyze(fullfile(folder, file));
    end

    function onSampleImage(~, ~)
        samplePath = findSampleImage(projectDir);
        if isempty(samplePath)
            uialert(fig, ['No sample fundus image found under data/. ' ...
                'Open an image from disk instead.'], 'No sample image');
            return
        end
        showAndAnalyze(samplePath);
    end

    function onAnalyze(~, ~)
        if isempty(state.image)
            uialert(fig, 'Open a fundus image first.', 'No image');
            return
        end
        runPrediction();
    end

    function showAndAnalyze(imagePath)
        image = imread(imagePath);
        if ndims(image) == 2
            image = repmat(image, 1, 1, 3);
        elseif size(image, 3) == 4
            image = image(:, :, 1:3);
        end
        state.image = image;
        state.result = [];
        imshow(image, 'Parent', axImage);
        title(axImage, sprintf('Original  (%s)', filenameOf(imagePath)));
        cla(axCam);
        title(axCam, 'Grad-CAM overlay');
        runPrediction();
    end

    function runPrediction()
        gradeLabel.Text = 'Analyzing…';
        drawnow;
        try
            result = predictIDRiDGrader(state.image, modelDir);
            state.result = result;
            banner.Text = modelStatusText(modelDir, result);
            renderResult(result);
        catch err
            gradeLabel.Text = 'Unable to grade image';
            uialert(fig, err.message, 'Unable to grade image');
        end
    end

    function renderResult(result)
        cla(axScores);
        if ~result.isUsable
            gradeLabel.Text = 'Recapture required';
            referLabel.Text = 'Referral: hold — ungradable image';
            expectedLabel.Text = 'Expected grade: —';
            qualityLabel.Text = 'Image quality: ungradable';
            metricsLabel.Text = '';
            noteLabel.Text = result.message;
            cla(axCam);
            text(axCam, 0.08, 0.5, 'No Grad-CAM until a usable image is captured.', ...
                'Units', 'normalized', 'Color', [0.3 0.3 0.3]);
            return
        end
        gradeLabel.Text = sprintf('Grade %d  ·  %s', ...
            result.grade, result.gradeName);
        if result.referable
            referLabel.Text = 'Referral: referable DR (grade ≥ 2)';
            referLabel.FontColor = [0.70 0.10 0.10];
        else
            referLabel.Text = 'Referral: not referable on this grade';
            referLabel.FontColor = [0.10 0.50 0.25];
        end
        expectedLabel.Text = sprintf('Expected grade %.2f  ·  reported grade %d', ...
            result.expectedGrade, result.grade);
        [modelProbability, modelIndex] = max(result.probabilities);

        modelGrade = modelIndex - 1;

        if modelGrade ~= result.grade
            noteLabel.Text = sprintf( ...
                ['Decoder override: neural network top class is Grade %d ' ...
                '(%.1f%%), while the frozen decoder reports Grade %d.'], ...
                modelGrade, ...
                100 * modelProbability, ...
                result.grade);
        else
            noteLabel.Text = sprintf( ...
                'Neural network top class agrees with reported Grade %d (%.1f%%).', ...
                result.grade, ...
                100 * modelProbability);
        end
        qualityLabel.Text = 'Image quality: gradable';
        metricsLabel.Text = '';
        noteLabel.Text = sprintf('%s %s', noteLabel.Text, result.message);
        drawScoreChart(axScores, result);
        showOverlay(result, alphaSlider.Value);
    end

    function onAlphaChanging(~, event)
        if ~isempty(state.result) && state.result.isUsable
            showOverlay(state.result, event.Value);
        end
    end

    function onAlphaChanged(~, event)
        if ~isempty(state.result) && state.result.isUsable
            showOverlay(state.result, event.Value);
        end
    end

    function showOverlay(result, alpha)

    % result.cam is produced by predictIDRiDGrader / computeGradCAM.
    % Improve display contrast without altering the underlying CAM.

    cam = double(result.cam);

    if isempty(cam)
        cla(axCam);
        text(axCam, 0.08, 0.5, ...
            'Grad-CAM unavailable.', ...
            'Units', 'normalized', ...
            'Color', [0.8 0.8 0.8], ...
            'FontSize', 14);
        return;
    end

    % Remove negative evidence for standard Grad-CAM visualization.
    cam = max(cam, 0);

    % Robust percentile normalization.
    % This prevents one extreme activation from making the rest
    % of the heatmap almost invisible.
    lo = prctile(cam(:), 20);
    hi = prctile(cam(:), 99);

    if hi > lo
        cam = (cam - lo) ./ (hi - lo);
    else
        cam = mat2gray(cam);
    end

    cam = min(max(cam, 0), 1);

    overlay = blendCamOverlay( ...
        state.image, ...
        cam, ...
        alpha);

    imshow(overlay, 'Parent', axCam);

    % Explain what class the current CAM represents.
    if isfield(result, 'camGrade')
        title(axCam, ...
            sprintf('Grad-CAM evidence — Grade %d', result.camGrade));
    else
        title(axCam, 'Grad-CAM evidence');
    end
end
end

function styleAxes(ax)
ax.XTick = [];
ax.YTick = [];
ax.Box = 'off';
ax.Color = [0.08 0.08 0.08];
end

function txt = modelStatusText(modelDir, result)
nFolds = countFoldModels(modelDir);
hasCal = isfile(fullfile(modelDir, 'cross_validation.mat'));
if nargin >= 2 && ~isempty(result) && isfield(result, 'demoMode') && result.demoMode
    txt = 'Demo mode: trained fold models or calibration were not found.';
    return
end
if nFolds > 0 && hasCal
    extra = '';
    if nargin >= 2 && ~isempty(result)
        if ~isempty(result.backboneName)
            extra = sprintf('  Backbone: %s.', result.backboneName);
        end
    end
    txt = sprintf('Using trained %d-fold ensemble under models/idrid_grade5/.%s', ...
        nFolds, extra);
else
    txt = ['No trained fold models yet — UI is in demo mode. ' ...
        'Run trainIDRiDGrader.m to replace placeholder scores.'];
end
end

function txt = qualityText(quality)
if isempty(quality)
    txt = '';
    return
end
if isfield(quality, 'isUsable') && ~quality.isUsable
    txt = 'Image quality: ungradable';
else
    txt = 'Image quality: gradable';
end
end

function drawScoreChart(axScores, result)
p = result.probabilities(:)';
if numel(p) < 5
    p(end+1:5) = 0;
end
cla(axScores);
bh = bar(axScores, 1:5, p, 0.72);
bh.FaceColor = 'flat';
bh.EdgeColor = 'none';
colors = repmat([0.18 0.62 0.66], 5, 1);
colors(result.grade + 1, :) = [0.92 0.48 0.12];
bh.CData = colors;
ymax = max(0.80, min(1, max(p) * 1.28 + 0.08));
axScores.YLim = [0 ymax];
axScores.XLim = [0.5 5.5];
axScores.XTick = 1:5;
axScores.XTickLabel = {'0','1','2','3','4'};
axScores.YLabel.String = 'Probability';
axScores.Box = 'off';
title(axScores, 'Grade probabilities (orange = reported grade)');
labelColor = [0.12 0.16 0.18];
try
    bg = axScores.Color;
    if isnumeric(bg) && mean(bg(:)) < 0.45
        labelColor = [0.93 0.95 0.96];
    end
catch
end
hold(axScores, 'on');
for i = 1:5
    text(axScores, i, p(i) + 0.035 * ymax, sprintf('%.0f%%', 100 * p(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, ...
        'FontWeight', 'bold', 'Color', labelColor);
end
hold(axScores, 'off');
end

function n = countFoldModels(modelDir)
n = 0;
for fold = 1:10
    if ~isfile(fullfile(modelDir, sprintf('fold-%d.mat', fold)))
        return
    end
    n = fold;
end
end

function name = filenameOf(path)
[~, name, ext] = fileparts(path);
name = [name ext];
end

function samplePath = findSampleImage(projectDir)
roots = { ...
    fullfile(projectDir, 'data', 'B. Disease Grading', 'B. Disease Grading', ...
        '1. Original Images', 'b. Testing Set'); ...
    fullfile(projectDir, 'data', 'B. Disease Grading', 'B. Disease Grading', ...
        '1. Original Images', 'a. Training Set'); ...
    fullfile(projectDir, 'data', 'A. Segmentation', 'A. Segmentation', ...
        '1. Original Images', 'b. Testing Set')};
samplePath = '';
for i = 1:numel(roots)
    files = dir(fullfile(roots{i}, '*.jpg'));
    if ~isempty(files)
        samplePath = fullfile(roots{i}, files(1).name);
        return
    end
end
end
