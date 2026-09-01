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

function [mrgnLkl,K] = CalculateMarginalLikelihood(x,y,theta)

% Calculates the marginal likelihood, given x, y and theta
% (hyperparameters)

% theta(1) : lambda
% theta(2) : l
% theta(3) : sigma

N = size(x,1);

% Covariance matrix given hyperparameters theta
K = AssembleCovariance(x,x, theta(1), theta(2)) + (theta(3)^2)*eye(N);

% Very naively:
%mrgnLkl = -0.5*y'*inv(K)*y - 0.5*log(det(K)) - length(x)*log*2*pi

% A bit less naively, using Cholesky factorization
L = chol(K,'lower');


% no factor 0.5 wtih Cholesky from LL' i.e. L^2 inside log -> 0.5 * 2 = 1
mrgnLkl = -0.5*y'*(L'\(L\y)) - sum(log(diag(L))) - 0.5*N*log(2*pi);


end

