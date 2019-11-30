function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

pre_sigmoid_dot_product = X*theta;
sigmoid_output = sigmoid(pre_sigmoid_dot_product);
J = 1/m * (-transpose(y)*log(sigmoid_output)-transpose(ones(m,1)-y)*log(ones(m,1)-sigmoid_output));
theta_no_bias = theta(2:end,1);
J += lambda/(2*m) * transpose(theta_no_bias)*theta_no_bias;

beta = sigmoid_output - y;
grad = (1/m)*transpose(X)*beta;
grad_tweaks = (lambda/m) * theta;
grad_tweaks(1) = 0;
grad += grad_tweaks;

end
