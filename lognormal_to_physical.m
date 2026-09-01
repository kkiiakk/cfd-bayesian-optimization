function [mean_phys, std_phys, mu_log, sigma_log] = ...
    lognormal_to_physical(mu_std, sigma_std, mean_log, std_log)

% Convert standardized log-output GP prediction back to log(NO_x)
mu_log = mean_log + std_log .* mu_std;

sigma_log = std_log .* sigma_std;


% If log(NO_x) is Gaussian, NO_x is lognormally distributed.
%
% Physical posterior mean:
mean_phys = exp(mu_log + 0.5 .* sigma_log.^2);


% Physical posterior variance and standard deviation:
var_phys = ...
    (exp(sigma_log.^2) - 1) .* ...
    exp(2 .* mu_log + sigma_log.^2);

std_phys = sqrt(max(var_phys,0));

end