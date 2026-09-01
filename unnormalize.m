function x = unnormalize(x_normalized, lb, ub)

x = lb + x_normalized .* (ub - lb);

end