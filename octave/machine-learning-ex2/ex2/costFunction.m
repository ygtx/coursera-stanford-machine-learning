function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%s

z = X * theta; % ?? I know theta is already transposed from horizontal to vertical though, ...
h = sigmoid(z);

size(h); % 100 x 1
size(log(h)); % 100 x 1

innerCostFunction1 = - y .* log( h ); % 100 x 1 .* 100 x 1 = 100 x 1
innerCostFunction2 = - ( ones(m, 1) .- y ) .* log( ones(m ,1) .- h ); % 100 x 1 
innerGrad = ( h .- y ) .* X;

J = (1 / m) .* sum( innerCostFunction1 .+ innerCostFunction2 );
grad = (1 / m) .* sum( innerGrad );


% =============================================================

end
