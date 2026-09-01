function result = find_constrained_optimum( ...
    x_training, ...
    y_training_NOx, theta_NOx, ...
    y_training_T, theta_T, ...
    mean_T, std_T, ...
    mean_NOx, std_NOx, ...
    lb, ub)

%% Find final continuous constrained GP optimum
%
% Objective:
%   minimize predicted physical NOx mean
%
% Constraint:
%   P(T_out >= 0.99*T_baseline) >= 0.95
%
% Optimization is performed in normalized input space [0,1]^3.


%% 1. Temperature constraint

T_baseline = 1618.73;
T_min = 0.95 * T_baseline;
p_min = 0.95;


%% 2. Generate Sobol points for multistart initialization

n_candidates = 512;
n_restarts = 12;

rng(1,'twister');

p = sobolset(3);
p = scramble(p,'MatousekAffineOwen');

X_candidates = net(p,n_candidates);


%% 3. Evaluate GP objective and feasibility at Sobol points

NOx_mean = zeros(n_candidates,1);
P_feas   = zeros(n_candidates,1);

for i = 1:n_candidates

    [NOx_mean(i), P_feas(i)] = ...
        evaluate_point(X_candidates(i,:));

end

%% Numerical scaling for fmincon objective

feasible_sobol = P_feas >= p_min;

if any(feasible_sobol)
    NOx_scale = min(NOx_mean(feasible_sobol));
else
    NOx_scale = min(NOx_mean);
end

% safeguard
NOx_scale = max(NOx_scale,1e-12);


%% 4. Select starting points

feasible = P_feas >= p_min;

if any(feasible)

    feasible_indices = find(feasible);

    % Rank feasible points by predicted NOx
    [~,order] = sort(NOx_mean(feasible_indices),'ascend');

    n_start = min(n_restarts,length(order));

    start_indices = feasible_indices(order(1:n_start));

else

    warning(['No Sobol candidate satisfies the probabilistic ', ...
             'temperature constraint. Starting from points with ', ...
             'highest feasibility probability.']);

    [~,order] = sort(P_feas,'descend');

    n_start = n_restarts;
    start_indices = order(1:n_start);

end

X_start = X_candidates(start_indices,:);


%% 5. Continuous constrained optimization

lb_norm = [0 0 0];
ub_norm = [1 1 1];

options = optimoptions( ...
    'fmincon', ...
    'Algorithm','sqp', ...
    'Display','none', ...
    'OptimalityTolerance',1e-8, ...
    'StepTolerance',1e-10, ...
    'ConstraintTolerance',1e-8, ...
    'MaxFunctionEvaluations',5000);


X_local = zeros(n_start,3);
NOx_local = zeros(n_start,1);
exitflag_local = zeros(n_start,1);

for j = 1:n_start

    x0 = X_start(j,:);

    [X_local(j,:), NOx_local(j), exitflag_local(j)] = ...
        fmincon( ...
            @objective_function, ...
            x0, ...
            [],[],[],[], ...
            lb_norm, ...
            ub_norm, ...
            @temperature_constraint, ...
            options);

end


%% 6. Check resulting solutions

P_local = zeros(n_start,1);

for j = 1:n_start
    [NOx_local(j),P_local(j)] = evaluate_point(X_local(j,:));
end

valid = (P_local >= p_min - 1e-6) & (exitflag_local > 0);

if ~any(valid)
    error('No valid constrained optimum was found.');
end


%% 7. Select best constrained solution

valid_indices = find(valid);

[~,local_best] = min(NOx_local(valid));

best_index = valid_indices(local_best);

x_opt_norm = X_local(best_index,:);

[NOx_opt, P_opt, ...
    std_NOx_opt, ...
    T_opt, std_T_opt] = ...
    evaluate_point(x_opt_norm);

x_opt_phys = ...
    lb + x_opt_norm .* (ub-lb);


%% 8. Store results

result.x_norm = x_opt_norm;
result.x_phys = x_opt_phys;

result.NOx_mean = NOx_opt;
result.NOx_std = std_NOx_opt;

result.T_mean = T_opt;
result.T_std = std_T_opt;

result.P_feas = P_opt;

result.T_min = T_min;
result.p_min = p_min;


%% 9. Display optimum

fprintf('\n========================================\n');
fprintf('FINAL CONTINUOUS CONSTRAINED GP OPTIMUM\n');
fprintf('========================================\n');

fprintf('v_air = %.6f m/s\n',x_opt_phys(1));
fprintf('v_FGR = %.6f m/s\n',x_opt_phys(2));
fprintf('T_FGR = %.2f K\n',x_opt_phys(3));

fprintf('\nPredicted NOx = %.12e\n',NOx_opt);
fprintf('NOx uncertainty = %.12e\n',std_NOx_opt);

fprintf('\nPredicted T = %.2f K\n',T_opt);
fprintf('T uncertainty = %.2f K\n',std_T_opt);

fprintf('P(T >= T_min) = %.6f\n',P_opt);
fprintf('Required probability = %.2f\n',p_min);
fprintf('T_min = %.2f K\n',T_min);


%% =========================================================
% Nested functions
%% =========================================================


    function f = objective_function(x)

        [NOx_phys,~] = evaluate_point(x);
    
        f = NOx_phys / NOx_scale;

    end


    function [c,ceq] = temperature_constraint(x)

        [~,~,~,mean_T_phys,std_T_phys] = evaluate_point(x);
    
        % One-sided Gaussian quantile corresponding to p_min
        z_req = norminv(p_min);
    
        % Equivalent to:
        % P(T_out >= T_min) >= p_min
        %
        % fmincon requires c(x) <= 0
        c = T_min - ...
            (mean_T_phys - z_req .* std_T_phys);
    
        ceq = [];

    end

    function [mean_NOx_phys, P, ...
              std_NOx_phys, ...
              mean_T_phys, std_T_phys] = ...
              evaluate_point(x)

        %% NOx GP

        [mu_NOx, Sigma_NOx] = ...
            GaussianProcessRegression( ...
                x_training, ...
                y_training_NOx, ...
                x, ...
                theta_NOx);

        std_NOx_std = ...
            sqrt(max(Sigma_NOx,0));

        [mean_NOx_phys, ...
         std_NOx_phys] = ...
            lognormal_to_physical( ...
                mu_NOx, ...
                std_NOx_std, ...
                mean_NOx, ...
                std_NOx);


        %% Temperature GP

        [mu_T, Sigma_T] = ...
            GaussianProcessRegression( ...
                x_training, ...
                y_training_T, ...
                x, ...
                theta_T);

        std_T_std = ...
            sqrt(max(Sigma_T,0));

        mean_T_phys = ...
            mean_T + std_T .* mu_T;

        std_T_phys = ...
            std_T .* std_T_std;

        std_T_safe = max(std_T_phys,1e-12);

        P = normcdf( ...
            (mean_T_phys - T_min) ./ std_T_safe);

    end

end