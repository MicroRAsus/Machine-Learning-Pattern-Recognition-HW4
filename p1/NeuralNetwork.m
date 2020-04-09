function [paramsFinal, MSE] = NeuralNetwork(X, Y, N, nPerceptrons)
close all,

% Determine/specify sizes of parameter matrices/vectors
nX = size(X,1);
nY = size(Y,1);
sizeParams = [nX;nPerceptrons;nY];

% Initialize model parameters with zeros
params.A = zeros(nPerceptrons,nX); % weight in first layer
params.b = zeros(nPerceptrons,1); % bias in first layer
params.C = zeros(nY,nPerceptrons);
params.d = mean(Y,2);
vecParamsInit = [params.A(:);params.b;params.C(:);params.d]; % vectorized parameters

% Optimize model using fminsearch, I could use forward and backward
% propagation and gradient descend as well.
options = optimset('MaxFunEvals', 100000, 'MaxIter', 100000);
vecParams = fminsearch(@(vecParams)(objectiveFunction(X,Y,sizeParams,vecParams)),vecParamsInit,options);

paramsFinal.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
paramsFinal.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
paramsFinal.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
paramsFinal.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
Yhat = mlpModel(X,paramsFinal);
MSE = lossMSE(Y, Yhat, N);
%fprintf('Final MSE loss: %.2f\n', lossMSE(Y, Yhat, N));


%
function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1); % size of input
nPerceptrons = sizeParams(2); % number of perceptron
nY = sizeParams(3); % size of output
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX); %get weight parameters of first layer from vectorized parameters using reshape
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons); %get bias parameters of first layer from vectorized parameters;
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params); % predicted value
objFncValue = lossMSE(Y, H, N); 
