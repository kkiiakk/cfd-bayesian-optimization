function improvement = expected_improvement_sampling(X_candidates, x_training, y_training_NOx, theta_NOx, y_training_T,theta_T,...
     mean_T, std_T, mean_NOx, std_NOx, yT, yNOx, lb, ub)

%% Constrained Expected Improvement sampling


% 2. GP predictions over candidate points

% NOx GP
[mu_NOx_candidates, Sigma_NOx_candidates] = ...
    GaussianProcessRegression( ...
        x_training, ...
        y_training_NOx, ...
        X_candidates, ...
        theta_NOx);

% Temperature GP
[mu_T_candidates, Sigma_T_candidates] = ...
    GaussianProcessRegression( ...
        x_training, ...
        y_training_T, ...
        X_candidates, ...
        theta_T);


% 3. Posterior standard deviations
% positive diagonal standard deviations

% Standardized NOx uncertainty
std_NOx_candidates = ...
    sqrt(max(diag(Sigma_NOx_candidates),0));

% Standardized temperature uncertainty
std_T_candidates = ...
    sqrt(max(diag(Sigma_T_candidates),0));


% 4. Temperature feasibility 

% Use baseline experiment as a probabilistic minimum temperature
% requirement
T_baseline = 1618.73;


% Minimum allowable outlet temperature: 95% of baseline
T_min = 0.95 * T_baseline;

% Required probability of satisfying temperature constraint
p_min = 0.95;

% Convert temperature GP prediction back to Kelvin
mu_T_candidates_phys = ...
    unstandardize(mu_T_candidates, mean_T, std_T);

std_T_candidates_phys = ...
    std_T .* std_T_candidates;


% Probability:
% P(T >= T_min)

std_T_safe = max(std_T_candidates_phys, 1e-12);

P_feas = normcdf( ...
    (mu_T_candidates_phys - T_min) ./ std_T_safe);



% 5. Determine current best FEASIBLE observed NOx value

% Convert standardized log(NO_x) GP prediction
% to physical NOx mean and standard deviation

[mu_NOx_candidates_phys, ...
 std_NOx_candidates_phys, ...
 mu_logNOx_candidates, ...
 std_logNOx_candidates] = ...
    lognormal_to_physical( ...
        mu_NOx_candidates, ...
        std_NOx_candidates, ...
        mean_NOx, ...
        std_NOx);

% Actual CFD observations satisfying temperature requirement 
% exactly (no probabilty)
feasible_training = yT >= T_min;

if any(feasible_training)

    % Best observed NOx among feasible CFD simulations
    y_best_phys = min(yNOx(feasible_training));

else

    % No feasible CFD point exists yet
    % In this case EI is not well-defined relative to a feasible incumbent.
    % Sample the point with highest probability of feasibility instead.

    fprintf('\nNo currently observed feasible CFD point.\n');
    fprintf('Selecting point with highest temperature feasibility probability.\n');

    acquisition = P_feas;

end


% 6. Expected Improvement for physical NOx

if any(feasible_training)

    % Improvement relative to current best observed feasible
    % physical NOx value
    improvement = ...
        y_best_phys - mu_NOx_candidates_phys;


    % Predictive standard deviation in log(NO_x) space
    sigma_log_safe = ...
        max(std_logNOx_candidates, 1e-12);


    % Standardized distance between the current best physical
    % NOx and the predicted log(NO_x) distribution
    z = ...
        (log(y_best_phys) - mu_logNOx_candidates) ...
        ./ sigma_log_safe;


    % Expected Improvement for a lognormal physical NOx prediction
    %
    % EI = y_best * Phi(z)
    %      - E[NOx] * Phi(z - sigma_log)

    % where E[NOx] = exp(mu_log + 0.5 .* sigma_log.^2) 

    EI_NOx = ...
        y_best_phys .* normcdf(z) ...
        - mu_NOx_candidates_phys .* ...
          normcdf(z - sigma_log_safe);


    % Correct deterministic limit when uncertainty -> 0
    nearly_zero = ...
        std_logNOx_candidates < 1e-12;

    EI_NOx(nearly_zero) = ...
        max(improvement(nearly_zero),0);


    % Numerical safeguard: EI cannot be negative
    EI_NOx = max(EI_NOx,0);

    % 7. CONSTRAINED Expected Improvement

    acquisition = EI_NOx .* P_feas;

end

% Save acquisition values for visualization before already evaluated
% points are set to -Inf
acquisition_plot = acquisition;


% 8. Avoid choosing an already evaluated CFD point
% based on squared distance of canditate to training point 

for i = 1:size(X_candidates,1)

    distances = sqrt( ...
        sum((x_training - X_candidates(i,:)).^2,2));

    if min(distances) < 1e-6
        acquisition(i) = -Inf;
    end

end


%% 9. Select next CFD simulation
% Multi-start continuous optimization of acquisition

n_restarts = 12;


% checking
[acq_sobol_max, idx_sobol_max] = max(acquisition);

x_sobol_best = X_candidates(idx_sobol_max,:);
x_sobol_best_phys = unnormalize(x_sobol_best,lb,ub);

fprintf('\nBest raw Sobol acquisition:\n');
fprintf('(for checking purposes before local optimization with restart)');
fprintf('Acquisition = %.12e\n',acq_sobol_max);
fprintf('v_air = %.6f\n',x_sobol_best_phys(1));
fprintf('v_FGR = %.6f\n',x_sobol_best_phys(2));
fprintf('T_FGR = %.2f\n',x_sobol_best_phys(3));

% Take best Sobol locations as starting points
[~,index_start] = maxk(acquisition,n_restarts);

X_start = X_candidates(index_start,:);

lb_norm = [0 0 0];
ub_norm = [1 1 1];

X_local = zeros(n_restarts,3);
acq_local = zeros(n_restarts,1);

options = optimoptions( ...
    'fmincon', ...
    'Algorithm','sqp', ...
    'Display','none');


% Scale acquisition for numerical optimization
acq_scale = max(acquisition(isfinite(acquisition)));

if acq_scale <= 0
    acq_scale = 1;
end


for j = 1:n_restarts

    x0 = X_start(j,:);

    [X_local(j,:),fval] = fmincon( ...
        @(x) -single_acquisition(x)/ acq_scale, ...
        x0, ...
        [],[],[],[], ...
        lb_norm,ub_norm, ...
        [],options);

    acq_local(j) = -fval* acq_scale;

end



% Best of the 12 continuous searches
[acquisition_max,best_restart] = max(acq_local);

x_next = X_local(best_restart,:);

x_next_phys = unnormalize(x_next,lb,ub);

[~, P_next, EI_next, ...
    mu_NOx_next_phys, std_NOx_next_phys, ...
    mu_T_next_phys, std_T_next_phys] = ...
    single_acquisition(x_next);


%% Visualize constrained Expected Improvement

% Convert all candidate inputs back to physical units
X_candidates_phys = unnormalize(X_candidates, lb, ub);

figure;

scatter3( ...
    X_candidates_phys(:,1), ...
    X_candidates_phys(:,2), ...
    X_candidates_phys(:,3), ...
    40, ...
    acquisition_plot, ...
    'filled');

hold on;

% Mark the point selected for the next CFD simulation
scatter3( ...
    x_next_phys(1), ...
    x_next_phys(2), ...
    x_next_phys(3), ...
    150, ...
    'filled');

xlabel('v_{air} [m/s]');
ylabel('v_{FGR} [m/s]');
zlabel('T_{FGR} [K]');

title('Constrained Expected Improvement');

colorbar;
grid on;


% 10. Display selected operating point

fprintf('\nNext CFD point from constrained Expected Improvement:\n');

fprintf('v_air = %.6f m/s\n', ...
    x_next_phys(1));

fprintf('v_FGR = %.6f m/s\n', ...
    x_next_phys(2));

fprintf('T_FGR = %.2f K\n', ...
    x_next_phys(3));

fprintf('\nAcquisition = %.6e\n', ...
    acquisition_max);

fprintf('P(T >= T_min) = %.6f\n', ...
    P_next);

fprintf('Predicted T = %.2f K\n', ...
    mu_T_next_phys);

fprintf('T uncertainty = %.2f K\n', ...
    std_T_next_phys);

fprintf('Predicted NOx = %.12e\n', ...
    mu_NOx_next_phys);

fprintf('NOx uncertainty = %.12e\n', ...
    std_NOx_next_phys);

if any(feasible_training)

    fprintf('Expected Improvement NOx = %.6e\n', ...
        EI_next);

end



%% Acquisition at one arbitrary continuous point
function [a, P, EI, ...
          mean_NOx_phys, std_NOx_phys, ...
          mean_T_phys, std_T_phys] = single_acquisition(x)

    % NOx GP
    [mu_NOx_single, Sigma_NOx_single] = ...
        GaussianProcessRegression( ...
            x_training, ...
            y_training_NOx, ...
            x, ...
            theta_NOx);

    % Temperature GP
    [mu_T_single, Sigma_T_single] = ...
        GaussianProcessRegression( ...
            x_training, ...
            y_training_T, ...
            x, ...
            theta_T);


    %% Temperature

    std_T_single = sqrt(max(Sigma_T_single,0));

    mean_T_phys = ...
        unstandardize(mu_T_single, mean_T, std_T);

    std_T_phys = ...
        std_T .* std_T_single;

    std_T_safe_single = max(std_T_phys,1e-12);

    P = normcdf( ...
        (mean_T_phys - T_min) ./ std_T_safe_single);


    %% NOx

    std_NOx_single = sqrt(max(Sigma_NOx_single,0));

    [mean_NOx_phys, ...
     std_NOx_phys, ...
     mu_log_single, ...
     sigma_log_single] = ...
        lognormal_to_physical( ...
            mu_NOx_single, ...
            std_NOx_single, ...
            mean_NOx, ...
            std_NOx);


    %% Expected Improvement

    if any(feasible_training)

        sigma_log_safe_single = ...
            max(sigma_log_single,1e-12);

        z_single = ...
            (log(y_best_phys) - mu_log_single) ...
            ./ sigma_log_safe_single;

        EI = ...
            y_best_phys .* normcdf(z_single) ...
            - mean_NOx_phys .* ...
              normcdf(z_single - sigma_log_safe_single);

        if sigma_log_single < 1e-12

            EI = max( ...
                y_best_phys - mean_NOx_phys, ...
                0);

        end

        EI = max(EI,0);

        a = EI .* P;

    else

        EI = 0;
        a = P;

    end

end


end

