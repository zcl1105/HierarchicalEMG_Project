function [Xz, mu, sigma] = zscore_safe(X)
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma == 0) = 1;
Xz = (X - mu) ./ sigma;
end
