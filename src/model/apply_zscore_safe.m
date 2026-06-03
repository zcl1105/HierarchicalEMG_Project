function Xz = apply_zscore_safe(X, mu, sigma)
sigma(sigma == 0) = 1;
Xz = (X - mu) ./ sigma;
end
