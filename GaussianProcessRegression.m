% GaussianProcessRegression: Computes the posterior mean and covariance of
% the Gaussian Process surrogate models 

% Cholesky factorization is used to solve the posterior mean and
% covariance without inverting the covariance matrix.

% Part of the CFD combustion optimization project with flue gas
% recirculation (FGR).
% Author: Kiia Kaaresvirta

% Parts of the final code edited using the following exercise 
% as initial template: 
%  Solution to problem sheet on Gaussian Processes
%
%  Lecture: Probability Theory and Uncertainty Quantification
%           Technical University of Munich
%


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


L = chol(K,'lower');
   
% Find distribution conditional on observations using Cholesky
% decomposition
alpha = L' \ (L \ y_training);
mu = K_s' * alpha;
V = L \ K_s;
Sigma = K_ss - V' * V;


end

