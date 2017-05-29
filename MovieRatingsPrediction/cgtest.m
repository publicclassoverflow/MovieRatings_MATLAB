% This program is used to calculate the best c and gamma values for different models
% Evaluations are based on the average sum of squared errors or accuracy
% The values are updated with ones that generate the smallest sse 
% and the greatest accuracy
% c and g are computed separately, with one of the two fixed while trying to 
% compute the optimal value of the other
clear
clc
load data
[num_user txt raw]=xlsread('user information.xls');
%% Quantify user's gender: M--1, F--2
num_user(strcmp(raw(:,3),'M'),3)=1;
num_user(strcmp(raw(:,3),'F'),3)=2;

% 1 - Creative art
num_user(strcmp(raw(:,4),'artist'),4)=1;
num_user(strcmp(raw(:,4),'writer'),4)=1;
num_user(strcmp(raw(:,4),'entertainment'),4)=1;
% 2 - Social works
num_user(strcmp(raw(:,4),'educator'),4)=2;
num_user(strcmp(raw(:,4),'librarian'),4)=2;
num_user(strcmp(raw(:,4),'healthcare'),4)=2;
num_user(strcmp(raw(:,4),'lawyer'),4)=2;
% 3 - Science
num_user(strcmp(raw(:,4),'scientist'),4)=3;
num_user(strcmp(raw(:,4),'programmer'),4)=3;
num_user(strcmp(raw(:,4),'engineer'),4)=3;
num_user(strcmp(raw(:,4),'doctor'),4)=3;
num_user(strcmp(raw(:,4),'technician'),4)=3;
% 4 - Business
num_user(strcmp(raw(:,4),'salesman'),4)=4;
num_user(strcmp(raw(:,4),'administrator'),4)=4;
num_user(strcmp(raw(:,4),'executive'),4)=4;
num_user(strcmp(raw(:,4),'marketing'),4)=4;
% 5 - Student
num_user(strcmp(raw(:,4),'student'),4)=5;
% 6 - Other
num_user(strcmp(raw(:,4),'homemaker'),4)=6;
num_user(strcmp(raw(:,4),'other'),4)=6;
num_user(strcmp(raw(:,4),'retired'),4)=6;
num_user(strcmp(raw(:,4),'none'),4)=6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate initial gueses for G and C values for testing
% gval = [1, 10, 30, 60, 100];
% gval = [0.1, 0.3, 0.6, 1];
% gval = [0.15, 0.2, 0.25, 0.3];
% gval = [5:15];
% gval = [1:5];
% gval = [2.0, 2.3, 3.0, 3.3, 4.0];
%% The best gamma value for RBF is 3
% gval = 3;
%% The best gamma value for sigmoid is 0.3
% gval = 0.3;
%% The best gamma value for classification is 0.3
gval = 0.3;

% cval = [1, 10, 30, 60, 100];
% cval = [0.01, 0.03, 0.1, 0.3, 0.5, 1];
% cval = [0.05, 0.075, 0.1, 0.125, 0.15];
% cval = [0.1, 0.2, 0.3, 0.4, 0.5];
% cval = [0.15, 0.175, 0.2, 0.225];
%% The best c value for classification is 0.2
cval = 0.2;
%% The best c value for RBF and sigmoid is 0.075
% cval = 0.075;

%% Run and produce models
% SVM classification

% gtestSSE = zeros(1, length(gval));
ctestSSE = zeros(1, length(cval));

% for k = 1:length(cval)
for k = 1:length(gval)
    sserr = [];
    for i=1:35
    % for i=1:size(movie,2)
        if length(movie(i).data) < 30
            continue;
        end

        temp = movie(i).data;
        data = [];
        for j=1:size(temp,1)
            data(j,1:3) = num_user(num_user(:,1) == temp(j,1),2:end);
            data(j,4) = temp(j,3);
        end
        % [rtrainResult, rtestResult, trainErr, testErr] = libsvm_regression(data, cval, gval(k));
        [rtrainResult, rtestResult, trainErr, testErr] = libsvm_regression(data, cval(k), gval);

        movie(i).trainError = trainErr(2);
        movie(i).testError = testErr(2);
        sserr = [sserr testErr(2)];
    end

    % gtestSSE(k) = mean(sserr);
    ctestSSE(k) = mean(sserr);
end

% for k = 1:length(cval)
% for k = 1:length(gval)
%     accuracy = [];
%     for i=1:35
%     % for i=1:size(movie,2)
%         if length(movie(i).data) < 30
%             continue;
%         end

%         temp = movie(i).data;
%         data = [];
%         for j=1:size(temp,1)
%             data(j,1:3) = num_user(num_user(:,1) == temp(j,1),2:end);
%             data(j,4) = temp(j,3);
%         end
%         [rtrainResult, rtestResult, trainAcc, testAcc] = libsvm_classify(data, cval, gval(k));
%         % [rtrainResult, rtestResult, trainAcc, testAcc] = libsvm_classify(data, cval(k), gval);

%         movie(i).trainAccuracy = trainAcc(1);
%         movie(i).testAccuracy = testAcc(1);
%         accuracy = [accuracy testAcc(1)];
%     end

%     gtestAcc(k) = mean(accuracy);
%     % ctestAcc(k) = mean(accuracy);
% end
