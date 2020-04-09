close all, clear all,
N=1000; n = 2; K=10; % N : num of samples; n = num of class; K : number of folds
Ntest = 10000;

% Generate samples
[x,l] = generateMultiringDataset(n, N); % x : data, 2x1000; l : label, 1x1000
[xTest, lTest] = generateMultiringDataset(n, Ntest);

% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','RBF','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
figure(2), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); % best box constraint and sigma
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','RBF','KernelScale',sigmaBest); % train gaussian svm with best hyperparameter and entire training set
d = SVMBest.predict(xTest')'; % Labels of testing data using the trained SVM
indINCORRECT = find(lTest.*d == -lTest); % Find testing samples that are incorrectly classified by the trained SVM
indCORRECT = find(lTest.*d == lTest); % Find testing samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(xTest(1,indCORRECT),xTest(2,indCORRECT),'g.'), hold on,
plot(xTest(1,indINCORRECT),xTest(2,indINCORRECT),'r.'), axis equal,
title('Testing Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/Ntest, % Empirical estimate probability of error 
fprintf('Testing data estimate probability of error: %.2f%%\n',pTrainingError*100);
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,

