function [trResult, teResult, trAcc, teAcc] = libsvm_classify(data, c, g)
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
%% Multiclass classification
cmd = [' -q -t 2 ', ' -c ', num2str(c), ' -g ', num2str(g)];
% cmd = [' -q -c ', num2str(c), ' -g ', num2str(g), ' -t 2 '];


mdl = svmtrain(trainLabel, trainMatrix, cmd);

%% SVM test
[trainPredict, trAcc] = svmpredict(trainLabel, trainMatrix, mdl, '-q');
[testPredict, teAcc] = svmpredict(testLabel, testMatrix, mdl, '-q');
trResult = [trainLabel trainPredict];
teResult = [testLabel testPredict];