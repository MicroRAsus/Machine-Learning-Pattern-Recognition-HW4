clear all, close all,

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';

Kvalues = [2, 3, 4, 5, 6, 7, 8, 9]; % desired numbers of clusters/components
F = 10; % number of fold for c.v.

for imageCounter = 1:size(filenames,2) % loop image
    imdata = imread(filenames{1,imageCounter}); %read image data 
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1), imshow(imdata);
    %length(size(imdata))==3 color image with RGB color values
    
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata); % row column depth
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
    
    GMModel = fitgmdist(x', 2, 'Options', options, 'RegularizationValue', 0.1); % fit x into 2 component GMM
    %LL = logLikelihood(x, GMModel.ComponentProportion, GMModel.mu', GMModel.Sigma);
    %disp(LL);
    labels = cluster(GMModel, x'); % classify data into 2 components by labeling it into 1,2
    labelImage = reshape(labels,R,C);
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1+1), imshow(uint8(labelImage*255/2));
    title(strcat({'K = '},num2str(2)));
    
    N = size(x,2); % number of pixels/samples;
    % Divide the data set into K approximately-equal-sized partitions
    dummy = ceil(linspace(0,N,F+1)); %linspace = evenly divide, ceil to get rid of decimal point
    for f = 1:F
        indPartitionLimits(f,:) = [dummy(f)+1,dummy(f+1)]; % save partion limit/index Fx2
    end
    
    shuffleInd = randperm(N); % shuffled index in x sample data, 1xN
    
    % Allocate space
    LLvalidate = zeros(F,length(Kvalues)); % F experiment x different component number
    AverageLLvalidate = zeros(1,length(Kvalues));
    for k = 1:length(Kvalues)
        K = Kvalues(k); % number of component
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
            GMModel = fitgmdist(xTrain', K, 'Options', options, 'RegularizationValue', 0.1); % fit gmm using training data
            LLvalidate(f, k) = logLikelihood(xValidate, GMModel.ComponentProportion, GMModel.mu', GMModel.Sigma); % evaluate model using validation data
        end
        AverageLLvalidate(1,k) = mean(LLvalidate(:,k)); % average validation log likelihood over folds
    end

    [~,indBestK] = max(AverageLLvalidate); % find best component number
    fprintf('Best number of clusters: %d\n', Kvalues(indBestK));
    GMModel = fitgmdist(x', Kvalues(indBestK), 'Options', options, 'RegularizationValue', 0.1);
    labels = cluster(GMModel, x');
    labelImage = reshape(labels,R,C);
    figure(2), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1+1), imshow(uint8(labelImage*255/Kvalues(indBestK)));
    title(strcat({'K = '},num2str(Kvalues(indBestK))));
end

%%
function logLH = logLikelihood(x, alpha, mu, Sigma)
    logLH = sum(log(evalGMM(x,alpha,mu,Sigma)));
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end