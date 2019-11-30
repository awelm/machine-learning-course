function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

g = arrayfun(@(x) 1/(1+e^(-x)), z)

end
