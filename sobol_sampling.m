% Number of initial CFD points
n = 9;

% Number of input variables
d = 3;

% Sobol sequence
p = sobolset(d);
p = scramble(p,'MatousekAffineOwen');

% Normalized samples in [0,1]^3
Xnorm = net(p,n);

% Bounds
lb = [0.35, 2.0, 373.15];    % [v_air, v_FGR, T_FGR]
ub = [0.65, 10, 1000];     % change T_FGR,max if needed

% Scale to physical range
X = lb + Xnorm .* (ub-lb);

% Display
disp(array2table(X,...
    'VariableNames',{'v_air','v_FGR','T_FGR'}))