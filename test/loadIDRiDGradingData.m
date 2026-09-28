function imds = loadIDRiDGradingData(baseDir, setType)
% LOADIDRIDGRADINGDATA  Build a labeled imageDatastore for IDRiD
% B.Disease Grading, matching each image to its DR grade from the CSV.
%
%   imds = loadIDRiDGradingData(baseDir, setType)
%
%   baseDir  - path to the extracted "B. Disease Grading" folder
%   setType  - 'train' or 'test'
%
% CONFIRMED CSV headers: "Image name", "Retinopathy grade", "Risk of
% macular edema". readtable sanitizes spaces to underscores by default,
% so these become Image_name / Retinopathy_grade in the loaded table.

    if strcmpi(setType, 'train')
        setFolder = 'a. Training Set';
        csvFile = fullfile(baseDir, 'Groundtruth', ...
            'IDRiD_Disease Grading_Training Labels.csv');
    elseif strcmpi(setType, 'test')
        setFolder = 'b. Testing Set';
        csvFile = fullfile(baseDir, 'Groundtruth', ...
            'IDRiD_Disease Grading_Testing Labels.csv');
    else
        error('setType must be ''train'' or ''test''');
    end

    if ~isfile(csvFile)
        error(['Groundtruth CSV not found at expected path: %s\n' ...
               'Check the exact folder/file name in your extracted zip.'], csvFile);
    end

    labelsTable = readtable(csvFile);

    imgCol = 'Image_name';
    gradeCol = 'Retinopathy_grade';

    varNames = labelsTable.Properties.VariableNames;
    if ~ismember(imgCol, varNames) || ~ismember(gradeCol, varNames)
        error(['Expected CSV columns not found. Actual columns are:\n%s\n' ...
               'Update imgCol/gradeCol in this function to match.'], ...
               strjoin(varNames, ', '));
    end

    imgDir = fullfile(baseDir, '1. Original Images', setFolder);

    imageNames = string(labelsTable.(imgCol));
    if ~endsWith(imageNames(1), '.jpg')
        imageNames = imageNames + ".jpg";
    end
    filePaths = fullfile(imgDir, imageNames);

    grades = categorical(labelsTable.(gradeCol));

    imds = imageDatastore(filePaths, 'Labels', grades);
end