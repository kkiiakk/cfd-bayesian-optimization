function x_std = standardize(x, x_mean, x_std)

x_std = (x-x_mean)./x_std;

end