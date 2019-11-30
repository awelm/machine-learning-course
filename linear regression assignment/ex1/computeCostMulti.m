function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

if (m!=0)
    predictions = X * theta;
    predictions_diff = predictions - y;
    predictions_diff_squared = predictions_diff.^2;
    J = sum(predictions_diff_squared) / (2*m);
endif

end
