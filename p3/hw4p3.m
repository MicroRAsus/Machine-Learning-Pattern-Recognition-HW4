function hw4p3(filename, K)
close all,

%filename = '3096_color.jpg';
%filename = '42049_color.jpg';

F = 10; % number of fold for c.v.
reg = 1e-2;

imdata = imread(filename); %read image data 
figure(1), subplot(1,2,1), imshow(imdata);
%length(size(imdata))==3 color image with RGB color values

[R,C,D] = size(imdata); N=R*C;%size if the sample/number of pixels
imdata = double(imdata); % row column depth
rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
for d = 1:D % append row index, column index, red, green, blue value into a 5-dim raw feature vector
    imdatad = imdata(:,:,d); % pick one color at a time
    features = [features;imdatad(:)']; 
end
minf = min(features,[],2); maxf = max(features,[],2);
ranges = maxf-minf;
x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
options = statset('MaxIter',300);

GMModel = fitgmdist(x', K, 'Options', options, 'RegularizationValue', reg); % fit x into K component GMM
%LL = logLikelihood(x, GMModel.ComponentProportion, GMModel.mu', GMModel.Sigma);
%disp(LL);
labels = cluster(GMModel, x'); % classify data into 2 components by labeling it into 1,2
labelImage = reshape(labels,R,C);
figure(1), subplot(1,2,2), imshow(uint8(labelImage*255/K));
title(strcat({'K = '},num2str(K)));

N = size(x,2); % number of pixels/samples;
% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,F+1)); %linspace = evenly divide, ceil to get rid of decimal point
for f = 1:F
    indPartitionLimits(f,:) = [dummy(f)+1,dummy(f+1)]; % save partion limit/index Fx2
end

shuffleInd = randperm(N); % shuffled index in x sample data, 1xN

% Allocate space
LLvalidate = zeros(F,1); % F experiment x different component number
AverageLLvalidate = 0;
for f = 1:F
    indValidate = [indPartitionLimits(f,1):indPartitionLimits(f,2)]; %index of validation partition
    xValidate = x(:,shuffleInd(indValidate)); % Using fold f in shuffled index as validation set
    if f == 1 % if first partition is validation
        indTrain = [indPartitionLimits(f,2)+1:N]; %use the rest of partition for training
    elseif f == F %if last partition is validation
        indTrain = [1:indPartitionLimits(f,1)-1]; % use 1 to start of the last partition -1 as training
    else
        indTrain = [indPartitionLimits(f-1,2)+1:indPartitionLimits(f+1,1)-1]; % use partition before and after the validation partition
    end
    xTrain = x(:,shuffleInd(indTrain)); % get training sample
    % Train model
    GMModel = fitgmdist(xTrain', K, 'Options', options, 'RegularizationValue', reg); % fit gmm using training data
    LLvalidate(f, 1) = logLikelihood(xValidate, GMModel.ComponentProportion, GMModel.mu', GMModel.Sigma); % evaluate model using validation data
end
AverageLLvalidate = mean(LLvalidate); % average validation log likelihood over folds

fprintf('Log liklihood of GMM component %d is: %.2f\n', K, AverageLLvalidate);

%%
function logLH = logLikelihood(x, alpha, mu, Sigma)
    logLH = sum(log(evalGMM(x,alpha,mu,Sigma)));

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);