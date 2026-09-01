function metrics = gp_parity_metrics( ...
    x_training, yNOx, yT, theta_NOx, theta_T)

%% GP parity plots and error metrics
% Compares CFD observations against GP posterior predictions
% at the training locations.
%
% Outputs:
%   R^2 and RMSE for NOx and outlet temperature
%   Parity plots with 95% predictive uncertainty intervals


%% 1. Recreate output transformations used for GP training

% NOx: log-transform, then standardize
yNOx_log = log(yNOx);

mean_NOx = mean(yNOx_log);
std_NOx  = std(yNOx_log);

y_training_NOx = ...
    (yNOx_log - mean_NOx) ./ std_NOx;


% Temperature: standardize directly
mean_T = mean(yT);
std_T  = std(yT);

y_training_T = ...
    (yT - mean_T) ./ std_T;


%% 2. GP predictions at CFD training locations

[mu_NOx_std, Sigma_NOx] = ...
    GaussianProcessRegression( ...
        x_training, ...
        y_training_NOx, ...
        x_training, ...
        theta_NOx);

[mu_T_std, Sigma_T] = ...
    GaussianProcessRegression( ...
        x_training, ...
        y_training_T, ...
        x_training, ...
        theta_T);


%% 3. Posterior standard deviations

sigma_NOx_std = ...
    sqrt(max(diag(Sigma_NOx),0));

sigma_T_std = ...
    sqrt(max(diag(Sigma_T),0));


%% 4. Convert temperature prediction back to Kelvin

mu_T_phys = ...
    mean_T + std_T .* mu_T_std;

sigma_T_phys = ...
    std_T .* sigma_T_std;


%% 5. Convert NOx GP from standardized log-space
%     back to physical NOx

[mu_NOx_phys, ...
 sigma_NOx_phys, ...
 mu_logNOx, ...
 sigma_logNOx] = ...
    lognormal_to_physical( ...
        mu_NOx_std, ...
        sigma_NOx_std, ...
        mean_NOx, ...
        std_NOx);


%% 6. Error metrics in PHYSICAL units

% NOx
residual_NOx = yNOx - mu_NOx_phys;

RMSE_NOx = sqrt( ...
    mean(residual_NOx.^2));

R2_NOx = 1 - ...
    sum(residual_NOx.^2) / ...
    sum((yNOx - mean(yNOx)).^2);


% Temperature
residual_T = yT - mu_T_phys;

RMSE_T = sqrt( ...
    mean(residual_T.^2));

R2_T = 1 - ...
    sum(residual_T.^2) / ...
    sum((yT - mean(yT)).^2);


%% 7. 95% uncertainty intervals

% NOx:
% Since log(NO_x) is Gaussian, use lognormal quantiles.
% This guarantees positive intervals.

z95 = 1.96;

NOx_lower = ...
    exp(mu_logNOx - z95 .* sigma_logNOx);

NOx_upper = ...
    exp(mu_logNOx + z95 .* sigma_logNOx);


% Temperature:
% Gaussian predictive interval

T_lower = ...
    mu_T_phys - z95 .* sigma_T_phys;

T_upper = ...
    mu_T_phys + z95 .* sigma_T_phys;


%% 8. Display metrics

fprintf('\n========================================\n');
fprintf('GP PARITY / TRAINING-FIT METRICS\n');
fprintf('========================================\n');

fprintf('\nNOx:\n');
fprintf('R^2  = %.6f\n', R2_NOx);
fprintf('RMSE = %.12e\n', RMSE_NOx);

fprintf('\nOutlet Temperature:\n');
fprintf('R^2  = %.6f\n', R2_T);
fprintf('RMSE = %.4f K\n', RMSE_T);


%% 9. Parity plots

figure;

tiledlayout(1,2);


%% NOx parity plot

nexttile;

% Asymmetric 95% uncertainty bars
errorbar( ...
    yNOx, ...
    mu_NOx_phys, ...
    mu_NOx_phys - NOx_lower, ...
    NOx_upper - mu_NOx_phys, ...
    'o', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.2, ...
    'CapSize',6);

hold on;


% Equality line
min_NOx = min([yNOx; NOx_lower]);
max_NOx = max([yNOx; NOx_upper]);

plot( ...
    [min_NOx max_NOx], ...
    [min_NOx max_NOx], ...
    '--', ...
    'LineWidth',1.2);


set(gca, ...
    'XScale','log', ...
    'YScale','log');

xlabel('CFD NO_x mass fraction');
ylabel('GP predicted NO_x mass fraction');

title(sprintf( ...
    'NO_x parity: R^2 = %.3f, RMSE = %.2e', ...
    R2_NOx, RMSE_NOx));

grid on;
axis square;


%% Temperature parity plot

nexttile;

errorbar( ...
    yT, ...
    mu_T_phys, ...
    mu_T_phys - T_lower, ...
    T_upper - mu_T_phys, ...
    'o', ...
    'LineStyle','none', ...
    'MarkerSize',7, ...
    'LineWidth',1.2, ...
    'CapSize',6);

hold on;


% Equality line
min_T_plot = min([yT; T_lower]);
max_T_plot = max([yT; T_upper]);

plot( ...
    [min_T_plot max_T_plot], ...
    [min_T_plot max_T_plot], ...
    '--', ...
    'LineWidth',1.2);


xlabel('CFD outlet temperature [K]');
ylabel('GP predicted outlet temperature [K]');

title(sprintf( ...
    'Temperature parity: R^2 = %.3f, RMSE = %.2f K', ...
    R2_T, RMSE_T));

grid on;
axis square;


sgtitle('Gaussian Process Parity Plots');


%% 10. Return results

metrics.R2_NOx   = R2_NOx;
metrics.RMSE_NOx = RMSE_NOx;

metrics.R2_T   = R2_T;
metrics.RMSE_T = RMSE_T;

metrics.NOx_prediction = mu_NOx_phys;
metrics.NOx_std        = sigma_NOx_phys;
metrics.NOx_lower95    = NOx_lower;
metrics.NOx_upper95    = NOx_upper;

metrics.T_prediction = mu_T_phys;
metrics.T_std        = sigma_T_phys;
metrics.T_lower95    = T_lower;
metrics.T_upper95    = T_upper;

end