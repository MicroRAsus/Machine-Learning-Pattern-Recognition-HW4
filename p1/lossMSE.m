function loss = lossMSE(Y, Yhat, N) %Y: ground truth, Yhat: prediction, N: # of sample
    loss = sum((Y-Yhat).*(Y-Yhat), 2) / N; % MSE - mean square error loss function - error between true value and predicted value
