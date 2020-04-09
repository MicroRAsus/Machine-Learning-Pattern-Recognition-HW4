close all, clear all,

N = 1000; K = 10; % N samples and K fold cross validation
numPerceptrons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; % different number of perceptrons
data = exam4q1_generateData(N);
x = data(1,:); %x1
y = data(2,:); %x2

Ntest = 10000; % generate testing set
dTest = exam4q1_generateData(Ntest);
xTest = dTest(1,:);
yTest = dTest(2,:); %x2

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1)); %linspace = evenly divide, ceil to get rid of decimal point
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; % save partion limit/index kx2
end

% Allocate space
MSEtrain = zeros(K,length(numPerceptrons)); % k experiment x different perceptron number
MSEvalidate = zeros(K,length(numPerceptrons)); 
AverageMSEtrain = zeros(1,length(numPerceptrons));
AverageMSEvalidate = zeros(1,length(numPerceptrons));

% Try all numnPerceptrons
for M = numPerceptrons
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)]; %index of validation partition
        xValidate = x(indValidate); % Using fold k as validation set
        yValidate = y(indValidate);
        if k == 1 % if first partition is validation
            indTrain = [indPartitionLimits(k,2)+1:N]; %use the rest of partition for training
        elseif k == K %if last partition is validation
            indTrain = [1:indPartitionLimits(k,1)-1]; % use 1 to start of the last partition -1 as training
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1]; % use partition before and after the validation partition
        end
        xTrain = x(indTrain); % get training sample
        yTrain = y(indTrain);
        Ntrain = length(indTrain); Nvalidate = length(indValidate); % # of training samples + # of validation samples
        % Train model parameters
        [wML,MSEtrain(k,M)] = NeuralNetwork(xTrain, yTrain, Ntrain, M);
        Yhat = mlpModel(xValidate,wML); % predict
        MSEvalidate(k,M) = lossMSE(yValidate,Yhat,size(yValidate, 2));
    end
    AverageMSEtrain(1,M) = mean(MSEtrain(:,M)); % average training MSE over folds
    AverageMSEvalidate(1,M) = mean(MSEvalidate(:,M)); % average validation MSE over folds
end

[~,indBestNumPerceptrons] = min(AverageMSEvalidate); % find best NumPerceptrons
fprintf('Best number of perceptrons: %d\n', numPerceptrons(indBestNumPerceptrons));

[wML,~] = NeuralNetwork(x, y, N, numPerceptrons(indBestNumPerceptrons)); % train model with best number of perceptrons and training data
Yhat = mlpModel(xTest,wML); % predict using trained model and test samples
testMSE = lossMSE(yTest,Yhat,size(yTest, 2)); % calculate test sample MSE
fprintf('Estimated mean squared error on testing data: %f\n',testMSE);

YhatTrain = mlpModel(x,wML); % predict using trained model and training samples
trainMSE = lossMSE(y,YhatTrain,N); % calculate test sample MSE
fprintf('Estimated mean squared error on training data: %f\n',trainMSE);

figure(1), clf
plot(xTest, yTest,'.g'); hold on,
plot(xTest, Yhat,'.r'); hold on,
xlabel('X_1'), ylabel('X_2'),
title('10,000 Test samples and best model prediction'),
legend('Original sample','Model prediction');

figure(2), clf
plot(x, y,'.g'); hold on,
plot(x, YhatTrain,'.r'); hold on,
xlabel('X_1'), ylabel('X_2'),
title('1,000 training samples and model prediction'),
legend('Original training sample','Model prediction');

figure(3), clf,
bar([AverageMSEtrain' AverageMSEvalidate']);
xlabel('# of perceptrons'); ylabel(strcat('MSE estimate with ',num2str(K),'-fold cross-validation'));
legend('Training MSE','Validation MSE');
title("MSE vs # of perceptrons");