function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
feature_count = size(theta)
grad = zeros(feature_count);
training_losses = size(m, 1)

for iter = 1:m
    x_i = X(iter, :);
    y_i = y(iter);
    z = x_i * theta;
    J = J - y_i  * log(sigmoid(z)) - (1-y_i) * log(1-sigmoid(z));
    training_losses(iter) = sigmoid(z) - y_i;
endfor

regularization_sum = 0;
for iter = 1:feature_count
    x_j = X(:, iter);
    grad(iter) = (training_losses * x_j)
    if(iter > 1)
        grad(iter) = grad(iter) + lambda*theta(iter);
        regularization_sum = regularization_sum + theta(iter)^2;
    endif
    grad(iter) = grad(iter)/m;
endfor

J = J + (lambda/2) * regularization_sum;
J = J/m;

end
