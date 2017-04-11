function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



predictions = X * theta;  % 12x2 * 2x1 = 12x1
errors = predictions .- y; % 12x1 - 12x1 = 12x1
squareErrors = errors .^ 2; 


J = 1/(2*m) * sum(squareErrors) + lambda/(2*m) * sum(theta(2:end,:) .^ 2);

innerGrad1 = (1/m) .* sum(errors .* X(:,[2:end]));
innerGrad2 = (lambda/m) .* theta([2:end],:);

grad([2:end],1) = innerGrad1' .+ innerGrad2;
grad(1,1) = ((1/m) .* sum(errors .* X(:,1)));

% errorsでは、j=0(octaveだとj=1になるけど)のデータを除外する理由なし
% コストに関しては普通にコスト関数から全thetaを使って計算した結果を使うって話

% =========================================================================

grad = grad(:);

end
