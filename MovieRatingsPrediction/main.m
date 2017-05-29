%{
    The main program to run this project
    At line 67 and 101, we deliberately skipped the movies with insufficient data
    Please comment out those lines if you want models based on all the movies
%}

clear
clc
load data
[num_user txt raw]=xlsread('user information.xls');
%% Quantify user's gender: M--1, F--2
num_user(strcmp(raw(:,3),'M'),3)=1;
num_user(strcmp(raw(:,3),'F'),3)=2;

%% Quantify user's occupations
% num_user(strcmp(raw(:,4),'administrator'),4)=1;
% num_user(strcmp(raw(:,4),'artist'),4)=2;
% num_user(strcmp(raw(:,4),'doctor'),4)=3;
% num_user(strcmp(raw(:,4),'educator'),4)=4;
% num_user(strcmp(raw(:,4),'engineer'),4)=5;
% num_user(strcmp(raw(:,4),'entertainment'),4)=6;
% num_user(strcmp(raw(:,4),'executive'),4)=7;
% num_user(strcmp(raw(:,4),'healthcare'),4)=8;
% num_user(strcmp(raw(:,4),'homemaker'),4)=9;
% num_user(strcmp(raw(:,4),'lawyer'),4)=10;
% num_user(strcmp(raw(:,4),'librarian'),4)=11;
% num_user(strcmp(raw(:,4),'marketing'),4)=12;
% num_user(strcmp(raw(:,4),'none'),4)=13;
% num_user(strcmp(raw(:,4),'other'),4)=14;
% num_user(strcmp(raw(:,4),'programmer'),4)=15;
% num_user(strcmp(raw(:,4),'retired'),4)=16;
% num_user(strcmp(raw(:,4),'salesman'),4)=17;
% num_user(strcmp(raw(:,4),'scientist'),4)=18;
% num_user(strcmp(raw(:,4),'student'),4)=19;
% num_user(strcmp(raw(:,4),'technician'),4)=20;
% num_user(strcmp(raw(:,4),'writer'),4)=21;

%% Group the users by their career field: 
%% art, science, social, etc
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
% 1st column - RBF; 2nd column - Classification
cval = [0.075 0.2];
gval = [3, 0.3];

%% Run and produce models
%% Regression
sserr = []; % Array to store sum of squared errors for all the movies
for i=1:size(movie,2)
    % Skip movies which has less than 30 ratings
    if length(movie(i).data) < 30
        continue;
    end
    temp = movie(i).data;
    data = [];
    for j=1:size(temp,1)
        data(j,1:3) = num_user(num_user(:,1) == temp(j,1),2:end);
        data(j,4) = temp(j,3);
    end
    [rtrResult, rteResult, trainErr, testErr] = libsvm_regression(data, cval(1), gval(1));
    movie(i).rtrainResult = rtrResult;
    movie(i).rtestResult = rteResult;
    movie(i).trainError = trainErr(2);
    movie(i).testError = testErr(2);
    sserr = [sserr testErr(2)];
end

% mean(sserr)
% max(sserr)
% min(sserr)

% Classification
accu = []; % Array to store accuracy for all the movies
for i=1:size(movie,2)
    % Skip movies which has less than 30 ratings
    if length(movie(i).data) < 30
        continue;
    end
    temp = movie(i).data;
    data = [];
    for j=1:size(temp,1)
        % Make input parameters
        data(j,1:3) = num_user(num_user(:,1) == temp(j,1),2:end);
        data(j,4) = temp(j,3);  % Get corresponding ratings
    end
    [ctrResult, cteResult, trainAcc, testAcc] = libsvm_classify(data, cval(2), gval(2));
    movie(i).ctrainResult = ctrResult;
    movie(i).ctestResult = cteResult;
    movie(i).trainAccuracy = trainAcc(1);
    movie(i).testAccuracy = testAcc(1);
    accu = [accu testAcc(1)];
end

% mean(accu)
