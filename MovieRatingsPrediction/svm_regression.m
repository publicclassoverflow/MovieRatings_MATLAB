function [trResult, teResult, trErr, teErr] = svm_regression(data)
[m,n] = size(data);
% Split dataset into training and testing data
trainMat = data(1:round(0.8*m),1:end-1);
trainLabel = data(1:round(0.8*m),end);

testMat = data(round(0.8*m)+1:end,1:end-1);
testLabel = data(round(0.8*m)+1:end,end);

%% Data normalization
[trainMatrix,PS] = mapminmax(trainMat');
trainMatrix = trainMatrix';
testMatrix = mapminmax('apply',testMat',PS);
testMatrix = testMatrix';

mdl = fitrsvm(trainMatrix, 'KernelFunction', 'RBF')

trainPredict = predict(mdl, trainLabel);
trResult = [trainLabel trainPredict];
testPredict = predict(mdl, testLabel);
teResult = [testLabel testPredict];
trErr = 0;
teErr = 0;