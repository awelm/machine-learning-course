function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = rows(X);
feature_count = columns(X);

mu = zeros(1, feature_count);
sigma = zeros(1, feature_count);

for iter = 1:feature_count
    feature_column = X(:, iter);
    feature_mean = mean(feature_column)
    meanless_feature_column = feature_column .- feature_mean;
    std_dev = std(meanless_feature_column);
    normalized_feature_column = meanless_feature_column ./ std_dev;

    X(:, iter) = normalized_feature_column;
    mu(iter) = feature_mean;
    sigma(iter) = std_dev;
end

X_norm = X;
fprintf("mean after norm\n");

end
