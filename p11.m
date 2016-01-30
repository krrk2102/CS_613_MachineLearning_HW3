clear;
rawData = csvread('CTG.csv', 2, 0);
rng(1);

testingCorrect = zeros(5, 50);
validatingCorrect = zeros(5, 50);
trainingCorrect =zeros(5, 50);
testingError = zeros(5, 50);
trainingError = zeros(5, 50);
validatingError = zeros(5, 50);

alpha = 0.01;
for foldNumber = 1 : 5
    
    while true
        rawTestingData = rawData(1+(foldNumber-1)*floor(size(rawData, 1)/5):foldNumber*floor(size(rawData, 1)/5), :);
        rawTrainingData = rawData;
        if foldNumber < 5
            rawValidatingData = rawData(foldNumber*floor(size(rawData,1)/5)+1:(foldNumber+1)*floor(size(rawData,1)/5), :);
            rawTrainingData((foldNumber-1)*floor(size(rawData,1)/5)+1:(foldNumber+1)*floor(size(rawData,1)/5), :) = [];
        else
            rawValidatingData = rawData(1:floor(size(rawData,1)/5)+1, :);
            rawTrainingData = rawData(floor(size(rawData,1)/5)+1:floor(size(rawData,1)*4/5), :);
        end
        classOneData = find(rawTrainingData(:, end) == 1);
        classTwoData = find(rawTrainingData(:, end) == 2);
        classThreeData = find(rawTrainingData(:, end) == 3);
        if isempty(classOneData) == false && isempty(classTwoData) == false && isempty(classThreeData) == false
            break;
        end
    end
    
    trainingData = [zscore(rawTrainingData(:, 1:21)) ones(size(rawTrainingData, 1), 1)];
    meanTrData = mean(rawTrainingData(:, 1:21)); 
    stdTrData = std(rawTrainingData(:, 1:21)); 
    
    for hSize = 1 : 50
        w1 = rand(size(trainingData, 2), hSize);
        w2 = rand(hSize, 3); 
        w1Newer = w1;
        w2Newer = w2;
        for iteration = 1 : 5 
            
            for i = 1 : size(trainingData, 1)
                
                hiddenLayerInput = trainingData(i, :) * w1;
                hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
                outputLayerInput = hiddenLayerOutput * w2;
                output = 1 ./ (1 + exp(-outputLayerInput));
                
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
        
        for i = 1 : size(trainingData, 1)
            
            trainingDataTuple = [(rawTrainingData(i, 1:21)-meanTrData)./stdTrData, 1];
            
            hiddenLayerInput = trainingDataTuple * w1;
            hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
            outputLayerInput = hiddenLayerOutput * w2;
            output = 1 ./ (1 + exp(-outputLayerInput));
            
            [~, classPrediction] = max(output);
            targetClass = rawTrainingData(i, end);
            if classPrediction == targetClass
                trainingCorrect(foldNumber, hSize) = trainingCorrect(foldNumber, hSize) + 1;
            end
        end
        
        for i = 1 : size(rawValidatingData, 1)
            
            validatingDataTuple = [(rawValidatingData(i, 1:21)-meanTrData)./stdTrData, 1];
            
            hiddenLayerInput = validatingDataTuple * w1;
            hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
            outputLayerInput = hiddenLayerOutput * w2;
            output = 1 ./ (1 + exp(-outputLayerInput));
            
            [~, classPrediction] = max(output);
            targetClass = rawValidatingData(i, end);
            if classPrediction == targetClass
                validatingCorrect(foldNumber, hSize) = validatingCorrect(foldNumber, hSize) + 1;
            end
        end
        
        for i = 1 : size(rawTestingData, 1)
            
            testingDataTuple = [(rawTestingData(i, 1:21)-meanTrData)./stdTrData, 1];
            
            hiddenLayerInput = testingDataTuple * w1;
            hiddenLayerOutput = 1 ./ (1 + exp(-hiddenLayerInput));
            outputLayerInput = hiddenLayerOutput * w2;
            output = 1 ./ (1 + exp(-outputLayerInput));
            
            [~, classPrediction] = max(output);
            targetClass = rawTestingData(i, end);
            if classPrediction == targetClass
                testingCorrect(foldNumber, hSize) = testingCorrect(foldNumber, hSize) + 1;
            end
        end
        
        testingError(foldNumber, hSize) = 1 - testingCorrect(foldNumber, hSize) / size(rawTestingData, 1);
        trainingError(foldNumber, hSize) = 1 - trainingCorrect(foldNumber, hSize) / size(trainingData, 1);
        validatingError(foldNumber,hSize) = 1 - validatingCorrect(foldNumber, hSize) / size(rawValidatingData, 1);
        display(foldNumber);
        display(hSize);
    end
end

trainingErrorAvg = mean(trainingError);
validatingErrorAvg = mean(validatingError);
testingErrorAvg = mean(testingError);

figure;
plot(1:50, trainingErrorAvg, 'r.-', 1:50, validatingErrorAvg, 'g.-', 1:50, testingErrorAvg, 'b-.');
legend('Traning Set Error', 'Valiadation Set Error', 'Testing Set Error');
