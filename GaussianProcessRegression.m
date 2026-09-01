%  Solution to problem sheet on Gaussian Processes
%
%  Lecture: Probability Theory and Uncertainty Quantification
%           Technical University of Munich
%
%
% DISCLAIMER: 
% The following code has been written for didactic purposes. As such the
% code is not properly vectorized and includes numeric operations which
% are generally not adviseable (e.g. explicit matrix inversions).


function [mu,Sigma] = GaussianProcessRegression(x_training,y_training,x_predict,theta, flag)

% Assembly
K = AssembleCovariance(x_training,x_training,theta(1),theta(2)) + (theta(3)^2)*eye(size(x_training,1));
K_s = AssembleCovariance(x_training, x_predict, theta(1), theta(2));

% this will yield covariance matrix of underlying noise-free function values
K_ss = AssembleCovariance(x_predict, x_predict, theta(1), theta(2));

% this will yield covariance matrix of noisy observed values
if nargin > 4
    K_ss = K_ss + (theta(3)^2)*eye(size(x_predict,1));
end
    
   
% Find distribution conditional on observations
mu = K_s'*inv(K)*y_training;                                  % in reality, one never takes the inverse : solve equation system, or use Cholesky decomposition
Sigma = K_ss - K_s'*inv(K)*K_s + 1e-8*eye(size(x_predict,1)); % we add diagonal matrix for numeric stabilization


end

