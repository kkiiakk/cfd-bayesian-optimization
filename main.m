%  Solution to problem sheet on Gaussian Processes
%
%  Lecture: Probability Theory and Uncertainty Quantification
%           Technical University of Munich
%
%
% DISCLAIMER: 
%
% The following code has been written for didactic purposes. As such the
% code is not properly vectorized and includes numeric operations which
% are generally not adviseable (e.g. explicit matrix inversions).


% This solution is used as a template for CFD Combustion with FGR
% optimization
clearvars;
clc;

run('CFD_Data.m');


% lower and upper bounds of data for normalization
% v_air, v_FGR, T_FGR
lb = [0.35, 2, 373.15];
ub = [0.65, 10, 1000];

N = size(yNOx);

%% Assemble covariance matrix

% optimized values for initial dataset (no baseline on 14/8)
% change for testing purposes to compare with different hyperparams
gamma_T = 2.096037;
l_T = 1.211859;
noise_stddev_T = 0.0;

gamma_NOx = 1.052918;
l_NOx = 0.410513;
noise_stddev_NOx = 0.000000040342;


x = normalize(X, lb, ub);


%% Visualize training data - NOx

visualize_training_data(X,yNOx);

title('CFD training data - outlet NOx');


%% Visualize training data - outlet temperature

visualize_training_data(X,yT);

title('CFD training data - outlet T');



%% create training data: normalized X, standardized yNOx and xT
x_training = x;

% standardize outputs
% Log-transform NOx to enforce positivity of physical predictions
yNOx_log = log(yNOx);

% Transformation parameters
mean_NOx = mean(yNOx_log);
std_NOx  = std(yNOx_log);

mean_T = mean(yT);
std_T  = std(yT);


% Standardize GP outputs
y_training_NOx = standardize(yNOx_log, mean_NOx, std_NOx);
y_training_T   = standardize(yT, mean_T, std_T);



%% Create 2D slice for GP visualization
% Fix one of the three input parameters for better visualization
v_air_fixed = 0.379563; 

% Grid for plotting (Sopol sampling bounds)
vFGR_plot = linspace(2, 10, 40);
TFGR_plot = linspace(373.15, 1000, 40);

% Create grid
[VFGR_grid, TFGR_grid] = ndgrid(vFGR_plot, TFGR_plot);

% Put input variables into a single matrix
x_slice = [
    v_air_fixed * ones(numel(VFGR_grid),1), ... 
    VFGR_grid(:), ...
    TFGR_grid(:)
];

% Normalize with the SAME bounds as training data
x_predict = normalize(x_slice, lb, ub);

Npredict = size(x_predict,1); % used for numerical stability


%% Perform Gaussian Process Regression

% GaussianProcessRegression adds noise to K training covariance
[mu_NOx, Sigma_NOx] = GaussianProcessRegression(x_training, y_training_NOx, x_predict,[gamma_NOx, l_NOx, noise_stddev_NOx]);
%L_NOx = chol(Sigma_NOx,'lower');


%% Gaussian process regression for T

[mu_T, Sigma_T] = GaussianProcessRegression(x_training, y_training_T, x_predict,[gamma_T, l_T, noise_stddev_T]);
%L_T = chol(Sigma_T,'lower');


%% Plot GP regression in physical units


%created a function for input/output tranformation to physical units
% not completed / running yet

%[VFGR_phys, TFGR_phys, mu_NOx_phys, mu_T_phys, std_NOx_phys, std_T_phys] = ...
%    transform_to_phys_units(VGR_grid, TFGR_grid, mu_NOx, mean_NOx, std_NOx,...
%    mu_T, mean_T, std_T, Sigma_NOx, Sigma_T)


% Transform prediction INPUTS back to physical units
x_predict_phys = unnormalize(x_predict, lb, ub);

VFGR_phys = reshape(x_predict_phys(:,2), size(VFGR_grid));
TFGR_phys = reshape(x_predict_phys(:,3), size(TFGR_grid));

% Temperature back to physical units
mu_T_phys = ...
    unstandardize(mu_T, mean_T, std_T);

std_T_phys = ...
    std_T .* sqrt(max(diag(Sigma_T),0));


% NOx: standardized log(NO_x) -> physical NOx
std_NOx_std = ...
    sqrt(max(diag(Sigma_NOx),0));

[mu_NOx_phys, std_NOx_phys, ...
 mu_logNOx_phys, sigma_logNOx_phys] = ...
    lognormal_to_physical( ...
        mu_NOx, ...
        std_NOx_std, ...
        mean_NOx, ...
        std_NOx);

%% Plot GP regression with NOISE

visualize_uncertainty(VFGR_grid, VFGR_phys, TFGR_phys, mu_NOx_phys, std_NOx_phys, mu_T_phys, std_T_phys)

sgtitle(['GP REGRESSIOn (with noise) at v_{air} = ', ...
    num2str(v_air_fixed), ' m/s']);


%% Marginal Likelihood, optimal hyperparameters


% Define marginal likelihood solely as a function of theta (given x,y)
% NOx training data
MarginalLikelihoodFct_NOx = @(theta) CalculateMarginalLikelihood(x_training,y_training_NOx, theta);

negLklFct_NOx = @(logtheta) ...
    -MarginalLikelihoodFct_NOx(exp(logtheta));

% Initial guess
logtheta0_NOx = log([1, 0.5, 0.1]);

% Optimization with fminsearch (optimize log parameters)
[logtheta_opt_NOx, negLkl_NOx] = ...
    fminsearch(negLklFct_NOx, logtheta0_NOx);

% Return from log-space
theta_NOx = exp(logtheta_opt_NOx);

gamma_NOx = theta_NOx(1);
l_NOx     = theta_NOx(2);
sigma_NOx = theta_NOx(3);


%T_FGR
MarginalLikelihoodFct_T = @(theta) CalculateMarginalLikelihood(x_training,y_training_T, theta);


negLklFct_T = @(logtheta) ...
    -MarginalLikelihoodFct_T(exp(logtheta));

% Initial guess
logtheta0_T = log([1, 0.5, 0.1]);

% Optimization
[logtheta_opt_T, negLkl_T] = ...
    fminsearch(negLklFct_T, logtheta0_T);

% Return from log-space
theta_T = exp(logtheta_opt_T);

gamma_T = theta_T(1);
l_T     = theta_T(2);
sigma_T = theta_T(3);




%% Display optimized hyperparameters

fprintf('\nOptimized NOx GP hyperparameters:\n');
fprintf('gamma   = %.6f\n', gamma_NOx);
fprintf('l       = %.6f\n', l_NOx);
fprintf('sigma_n = %.12f\n', sigma_NOx);
fprintf('log marginal likelihood = %.6f\n', -negLkl_NOx);

fprintf('\nOptimized Temperature GP hyperparameters:\n');
fprintf('gamma   = %.6f\n', gamma_T);
fprintf('l       = %.6f\n', l_T);
fprintf('sigma_n = %.12f\n', sigma_T);
fprintf('log marginal likelihood = %.6f\n', -negLkl_T);



%% Perform Gaussian Process Regression with optimized hyperparameters

% NOx GP
[mu_NOx, Sigma_NOx] = GaussianProcessRegression(x_training, y_training_NOx, x_predict, theta_NOx);


% Temperature GP
[mu_T, Sigma_T] = GaussianProcessRegression(x_training, y_training_T,x_predict, theta_T);


%% Final GP plots using optimized hyperparameters

% Transform prediction inputs back to physical units

x_predict_phys = unnormalize(x_predict, lb, ub);

VFGR_phys = reshape(x_predict_phys(:,2), size(VFGR_grid));

TFGR_phys = reshape(x_predict_phys(:,3), size(TFGR_grid));


% Temperature back to physical units
mu_T_phys = ...
    unstandardize(mu_T, mean_T, std_T);

std_T_phys = ...
    std_T .* sqrt(max(diag(Sigma_T),0));


% NOx: standardized log(NO_x) -> physical NOx
std_NOx_std = ...
    sqrt(max(diag(Sigma_NOx),0));

[mu_NOx_phys, std_NOx_phys] = ...
    lognormal_to_physical( ...
        mu_NOx, ...
        std_NOx_std, ...
        mean_NOx, ...
        std_NOx);


%% Final plots with optimized parameters

visualize_uncertainty(VFGR_grid, VFGR_phys, TFGR_phys, mu_NOx_phys, std_NOx_phys, mu_T_phys, std_T_phys)
sgtitle(['Optimized GP models at v_{air} = ', ...
    num2str(v_air_fixed), ' m/s']);


%% Expected Improvement Sampling
% Raw Sobol points for acquisition optimization

n_candidates = 512;

% Fixed random seed -> reproducible Sobol scrambling
rng(1,'twister');

% Sobol points directly in normalized design space [0,1]^3
p = sobolset(3);
p = scramble(p,'MatousekAffineOwen');

X_candidates = net(p,n_candidates);

% Physical values only for visualization
X_candidates_phys = unnormalize(X_candidates,lb,ub);

expected_improvement_sampling( ...
    X_candidates, x_training, ...
    y_training_NOx, theta_NOx, ...
    y_training_T, theta_T, ...
    mean_T, std_T, mean_NOx, std_NOx, ...
    yT, yNOx, lb, ub);



metrics = gp_parity_metrics( ...
    x_training, ...
    yNOx, ...
    yT, ...
    theta_NOx, ...
    theta_T);



result_opt = find_constrained_optimum( ...
    x_training, ...
    y_training_NOx, theta_NOx, ...
    y_training_T, theta_T, ...
    mean_T, std_T, ...
    mean_NOx, std_NOx, ...
    lb, ub);




%% Optimum CFD validation figure

case_labels = { ...
    'OS1', ...
    'OS1.1', ...
    'OS1.2', ...
    'OS1.3'};


%% Historical GP NOx predictions

NOx_pred_OS = [
    3.335190e-8
    1.629810e-8
    3.039058e-8
    1.728196e-8
];


%% Historical GP physical NOx uncertainty (standard deviation)

NOx_std_OS = [
    2.57342e-9
    3.34869e-9
    1.79197e-9
    1.39418e-9
];


%% Actual CFD NOx

NOx_CFD_OS = [
    3.483308e-8
    8.893065e-9
    2.958364e-8
    1.666769e-8
];


%% Historical GP outlet-temperature predictions

T_pred_OS = [
    1546.280
    1553.670
    1543.600
    1539.810
];


%% Historical GP temperature uncertainty (standard deviation)

T_std_OS = [
    5.16
    9.66
    3.53
    1.23
];


%% Actual CFD outlet temperatures

T_CFD_OS = [
    1547.241
    1510.308
    1552.758
    1537.540
];


%% Temperature constraint

T_baseline = 1618.73;

T_min = 0.95 * T_baseline;


%% Plot

OS_validation = ...
    plot_optimum_validation( ...
        case_labels, ...
        NOx_pred_OS, ...
        NOx_std_OS, ...
        NOx_CFD_OS, ...
        T_pred_OS, ...
        T_std_OS, ...
        T_CFD_OS, ...
        T_min);