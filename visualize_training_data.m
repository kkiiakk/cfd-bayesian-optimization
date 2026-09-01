function scatter =visualize_training_data(datax, datay)
% datax: 3 column matrixes of input1 v_air,input 2 v_FGR and input 3 t_FGR
% datay: 1 column of output data
figure;


%v_air, v_FGR, T_FGR, markersize, y_data, style
scatter3(datax(:,1), datax(:,2), datax(:,3), ...
         80, datay, 'filled');

xlabel('v_{air} [m/s]');
ylabel('v_{FGR} [m/s]');
zlabel('T_{FGR} [K]');

colorbar;
grid on;


end