%AssembleCovariance: Assembles the squared exponential covariance matrix

% Part of the CFD combustion optimization project with flue gas
% recirculation (FGR).
% Author: Kiia Kaaresvirta

% Parts of the final code edited using the following exercise 
% as initial template: 
%  Solution to problem sheet on Gaussian Processes
%
%  Lecture: Probability Theory and Uncertainty Quantification
%           Technical University of Munich


function C = AssembleCovariance(x1,x2,gamma,l)

% Assemble covariance matrix using Gaussian covariance function
%
% x1: N x D matrix
% x2: M x D matrix
%
% Each ROW represents one input point.
% Each COLUMN represents one input dimension.
%
% For this project:
% D = 3:
% [v_air, v_FGR, T_FGR]

N = size(x1,1);
M = size(x2,1);

C = zeros(N,M);

% Gaussian / squared exponential covariance function
k = @(a,b) (gamma^2) * ...
    exp(-0.5 * (1/(l^2)) * sum((a-b).^2));

for n = 1:N

    for m = 1:M

        C(n,m) = k(x1(n,:),x2(m,:));

    end

end

end
