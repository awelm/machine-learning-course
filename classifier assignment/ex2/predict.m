function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

p = X * theta;
p = sigmoid(p);
for iter = 1:m
    if(p(iter) >= 0.5)
        p(iter) = 1;
    else
        p(iter) = 0;
    endif
endfor

end