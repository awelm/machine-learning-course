function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

J = 0;

if (m!=0)
    predictions = X * theta;
    predictions_diff = predictions - y;
    predictions_diff_squared = predictions_diff.^2;
    J = sum(predictions_diff_squared) / (2*m);
endif

end
