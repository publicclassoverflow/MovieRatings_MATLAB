function [trResult, teResult, trErr, teErr] = libsvm_regression(data, c, g)
[m,n] = size(data);
% Split dataset into training and testing data
trainMat = data(1:round(0.8*m),1:end-1);
trainLabel = data(1:round(0.8*m),end);

testMat = data(round(0.8*m)+1:end,1:end-1);
testLabel = data(round(0.8*m)+1:end,end);

% Data normalization
[trainMatrix,PS] = mapminmax(trainMat');
trainMatrix = trainMatrix';
testMatrix = mapminmax('apply',testMat',PS);
testMatrix = testMatrix';

%% SVM building/training (RBF kernel function)
%% Regression
% RBF
cmd = [' -q -s 3 ', ' -t 2 ', ' -c ', num2str(c), ' -g ', num2str(g)];

% Sigmoid
% cmd = [' -q -s 3 ', ' -t 3 ', ' -c ' num2str(c), ' -g ', num2str(g)];

% Polynomial
% cmd = [' -q -s 3 ', ' -t 1 ', ' -c ' num2str(c), ' -g ', num2str(g)];

% Linear
% cmd = [' -q -s 3 ', ' -t 0 ', ' -c ' num2str(c)];

mdl = svmtrain(trainLabel, trainMatrix, cmd);

%% SVM test
% Quiet mode enabled
[trainPredict, trErr] = svmpredict(trainLabel, trainMatrix, mdl, '-q');
[testPredict, teErr] = svmpredict(testLabel, testMatrix, mdl, '-q');

% Quiet mode disabled
% [trainPredict, trErr] = svmpredict(trainLabel, trainMatrix, mdl);
% [testPredict,teErr] = svmpredict(testLabel, testMatrix, mdl);

trResult = [trainLabel trainPredict];
teResult = [testLabel testPredict];