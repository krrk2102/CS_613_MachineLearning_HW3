clear;
rawData = csvread('CTG.csv', 2, 0);
rng(1);

randomIndex = randperm(size(rawData, 1));
rawTrainingData = rawData(randomIndex(1: floor(size(rawData, 1)/2)), :);
rawTestingData = rawData(randomIndex(floor(size(rawData, 1)/2)+1 : end), :);
trainingData = [zscore(rawTrainingData(:, 1:21)), ones(size(rawTrainingData, 1), 1)];

alpha = 0.01; 
hSize = 20;
w1 = rand(size(trainingData, 2), hSize);
w2 = rand(hSize, 3);
w1Newer = rand(size(trainingData, 2), hSize);
w2Newer = rand(hSize, 3);

for iteration = 1 : 20 
    for i = 1 : size(trainingData, 1)
        
        hiddenLayerInput = trainingData(i, :) * w1;
        hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
        OutputLayerInput = hiddenLayerOutput * w2;
        output = 1 ./ (1 + exp(-OutputLayerInput));
        
        targetClass = rawTrainingData(i, end);
        targetValues = zeros(1, 3);
        targetValues(targetClass) = 1;
        
        deltaOutput = zeros(1, 3);
        for j = 1 : 3
            deltaOutput(j) = (targetValues(j) - output(j)) * output(j) * (1 - output(j));
            w2Newer(:, j) = w2(:, j) + alpha * deltaOutput(j) * hiddenLayerOutput';
        end
        deltaHiddenLayer = zeros(1, hSize);
        for h = 1 : hSize
            deltaHiddenLayer(h) = hiddenLayerOutput(h) * (1 - hiddenLayerOutput(h)) * (w2Newer(h, :) * deltaOutput');
            w1Newer(:, h) = w1(:, h) + alpha * deltaHiddenLayer(h) * trainingData(i, :)';
        end
        w1 = w1Newer;
        w2 = w2Newer;
    end
end

testingData = rawTestingData(:, 1:21);
testingCorrect = 0;
meanTrData = mean(rawTrainingData(:, 1:21)); 
stdTrData = std(rawTrainingData(:, 1:21)); 

for i = 1 : size(testingData, 1)
    
    testingDataTuple = [(testingData(i, :)-meanTrData)./stdTrData, 1];
    hiddenLayerInput = testingDataTuple * w1;
    hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
    OutputLayerInput = hiddenLayerOutput * w2;
    output = 1 ./ (1 + exp(-OutputLayerInput));
    
    [~, classPrediction] = max(output);
    targetClass = rawTestingData(i, end);
    if classPrediction == targetClass
        testingCorrect = testingCorrect + 1;
    end
end

testingError = 1 - testingCorrect / size(testingData, 1);

    
        
    
    
    






