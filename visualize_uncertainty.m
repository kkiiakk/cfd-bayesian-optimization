function visualize_uncertainty( ...
    VFGR_grid, VFGR_phys, TFGR_phys, ...
    mu_NOx_phys, std_NOx_phys, ...
    mu_T_phys, std_T_phys)

figure('Color','w');

tiledlayout(2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');


%% =========================================================
% 1. NOx mean
%% =========================================================

ax1 = nexttile;

% Show orders of magnitude directly
log10_NOx_mean = ...
    log10(max(mu_NOx_phys,1e-20));

contourf( ...
    VFGR_phys, ...
    TFGR_phys, ...
    reshape(log10_NOx_mean,size(VFGR_grid)), ...
    20, ...
    'LineColor','none');

cb1 = colorbar;

cb1.Label.String = ...
    'log_{10}(NO mass fraction)';

xlabel('v_{FGR} [m/s]');
ylabel('T_{FGR} [K]');

title('NO predicted mean');

grid on;
box on;


%% =========================================================
%% NOx relative uncertainty

ax2 = nexttile;

NOx_relative_uncertainty = ...
    100 .* std_NOx_phys ./ max(mu_NOx_phys,1e-20);

contourf( ...
    VFGR_phys, ...
    TFGR_phys, ...
    reshape(NOx_relative_uncertainty,size(VFGR_grid)), ...
    20, ...
    'LineColor','none');

cb2 = colorbar(ax2);
cb2.Label.String = 'Relative NO uncertainty [%]';

xlabel('v_{FGR} [m/s]');
ylabel('T_{FGR} [K]');
title('NO GP relative uncertainty');

grid on;
box on;

%% =========================================================
% 3. Temperature mean
%% =========================================================

ax3 = nexttile;

contourf( ...
    VFGR_phys, ...
    TFGR_phys, ...
    reshape(mu_T_phys,size(VFGR_grid)), ...
    20, ...
    'LineColor','none');

cb3 = colorbar;

cb3.Label.String = ...
    'Outlet temperature [K]';

xlabel('v_{FGR} [m/s]');
ylabel('T_{FGR} [K]');

title('Outlet temperature predicted mean');

grid on;
box on;


%% =========================================================
% 4. Temperature uncertainty
%% =========================================================

ax4 = nexttile;

contourf( ...
    VFGR_phys, ...
    TFGR_phys, ...
    reshape(std_T_phys,size(VFGR_grid)), ...
    20, ...
    'LineColor','none');

cb4 = colorbar;

cb4.Label.String = ...
    '\sigma_T [K]';

xlabel('v_{FGR} [m/s]');
ylabel('T_{FGR} [K]');

title('Outlet temperature GP uncertainty');

grid on;
box on;

end