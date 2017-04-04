function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);

innerCostFunction11 = - y .* log( h ); % 100 x 1 .* 100 x 1 = 100 x 1
innerCostFunction12 = - ( ones(m, 1) .- y ) .* log( ones(m ,1) .- h ); % 100 x 1 
innerCostFunction2 = (lambda / (2 * m)) .* sum(theta .^ 2 );

J = (1 / m) .* sum( innerCostFunction11 .+ innerCostFunction12 ) .+ innerCostFunction2;



innerGrad1 = (h .- y) .* X(:, [2:end]);
innerGrad2 = (lambda / m) .* theta([2:end], :);
grad([2:end], 1) = ((1 / m) .* sum( innerGrad1 )) .+ innerGrad2';

grad(1, 1) = (1 / m) .* sum( (h .- y) .* X(:, 1) );


% =============================================================

end
