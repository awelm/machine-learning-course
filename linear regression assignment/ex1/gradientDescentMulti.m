function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
feature_count = length(theta)
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    predictions = X * theta;
    predictions_diff = predictions - y;
    theta_diffs = zeros(feature_count, 1);
    for j = 1:feature_count
        theta_diffs(j) = (alpha/m) * transpose(predictions_diff) * X(:, j);
    end
    theta -= theta_diffs;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
