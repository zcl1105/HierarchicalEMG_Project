function xFiltered = my_notch(f0, fs, x)
ts = 1 / fs;
w0 = 2 * pi * f0 * ts;
alpha = -2 * cos(w0);
beta = 0.96;

b = [1, alpha, 1];
a = [1, alpha * beta, beta^2];

xFiltered = filter(b, a, x);
end
