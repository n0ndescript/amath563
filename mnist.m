%%Read Images
[imagesTrain, labelsTrain] = mnist_parse(...
                       '/Users/svijayakrishnan/Downloads/train-images-idx3-ubyte',... 
                       '/Users/svijayakrishnan/Downloads/train-labels-idx1-ubyte');

[imagesTest, labelsTest] = mnist_parse(...
                       '/Users/svijayakrishnan/Downloads/t10k-images-idx3-ubyte',... 
                       '/Users/svijayakrishnan/Downloads/t10k-labels-idx1-ubyte');
%Reshape image data to a single column
imagesTrain = double(reshape(imagesTrain, size(imagesTrain,1)*size(imagesTrain,2), []).');
imagesTest = double(reshape(imagesTest, size(imagesTest,1)*size(imagesTest,2), []).');

%% One hot Encoding
oneHotTrain = oneHotEncode(labelsTrain);
oneHotTest = oneHotEncode(labelsTest);



%%
xBackslash = imagesTrain \ oneHotTrain;
%%
[numErrorsBackslash, labelsPredictedBackslash] = tally(imagesTest, oneHotTest, xBackslash, 0.5);

%% Split into per digit datasets 
numErrorsBackslashDigit = zeros(10);
for i = 1:10
    imagesTrainDigit= imagesTrain(find(labelsTrain == i),:);
    oneHotTrainDigit = oneHotTrain(find(labelsTrain == i),:);
    imagesTestDigit= imagesTest(find(labelsTest == i),:);
    oneHotTestDigit = oneHotTest(find(labelsTest == i),:);
    xBackSlashDigit = imagesTrainDigit \ oneHotTrainDigit;
    [numErrorsBackslashDigit(i), ~] = tally(imagesTestDigit, oneHotTestDigit, xBackslash, 0.5);
end
%%
xLasso = zeros(784,10);
for i = 1:10
    x = lasso(imagesTrain,oneHotTrain(:,i),'Lambda', 0.1);
    xLasso(:,i) = x;
end
%%

[numErrorsLasso, labelsPredictedLasso] = tally(imagesTest, oneHotTest, xLasso, 0.05);

%%
XRidge = zeros(784,10);
for i = 1:10
    x = ridge(oneHotTrain(:,i),imagesTrain, 0.1);
    xRidge(:,i) = x;
end
%%
[numErrorsRidge, labelsPredictedRidge] = tally(imagesTest, oneHotTest, xRidge, 50);

%%
xPinv = pinv(imagesTrain) * oneHotTrain;

[numErrorsPinv, labelsPredictedPinv] = tally(imagesTest, oneHotTest, xPinv, 0.5);

%%
function [oneHot] = oneHotEncode(labels)
    numImages = size(labels,1);
    oneHot = double(zeros(numImages, 10));
    for i = 1: numImages
        pos = labels(i);
        if pos == 0
            pos = 10;
        end
        oneHot(i, pos) = 1;
    end
end

function [numErrors, labelsPredicted] = tally(imagesTest,oneHotBTest, xVec, tol)
    oneHotPredicted = imagesTest * xVec;
    indices = abs(oneHotPredicted) < tol;
    oneHotPredicted(indices) = 0;
    indices = abs(oneHotPredicted) >= tol;
    oneHotPredicted(indices) = 1;
    errors = oneHotPredicted - oneHotBTest; 
    
    labelsPredicted = zeros(size(oneHotPredicted, 2),1);
    for i = 1:size(oneHotPredicted, 2)
       labels = find(oneHotPredicted(i,:) > 0);
       if (size(labels) > 0) 
        labelsPredicted(i) = labels(1);
       end
    end
    numErrors = size(find(abs(errors) > 0),1);
end
                   
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - Images: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');

    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid2, totalImages, 'uint8');

% Close the files
fclose(fid1);
fclose(fid2);

end
