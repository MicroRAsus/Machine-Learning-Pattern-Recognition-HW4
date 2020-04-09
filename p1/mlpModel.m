% single layer neural net model
function H = mlpModel(X,params)
N = size(X,2);                          % number of samples
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = softPlusActFunc(U);              % z, value after non-linearity, num of perceptron x N
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations


% activation function
function out = softPlusActFunc(in)
out = log(1+exp(in)); % softplus activation