function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   /num[J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25x401
Theta2_grad = zeros(size(Theta2)); % 10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% --------------------- frontpropagation ---------------------

X = [ones(size(X, 1), 1) X]; % 5000x401
z_l2 = X * Theta1'; % 5000x401 * 25x401' = 5000x25
A_l2 = 1.0 ./ (1.0 .+ exp(-z_l2)); % g(z) = 1/(1+exp(-z)) and z = X*Theta1' 
% 5000x25

A_l2 = [ones(size(A_l2,1), 1) A_l2]; % 5000x26
z_l3 = A_l2 * Theta2'; % 5000x26 * 10x26' = 5000x10
A_l3 = 1.0 ./ (1.0 .+ exp(-z_l3)); % g(z) = 1/(1+exp(-z)) and z = X*Theta2'
% 5000x10

y_row_count = size(y, 1);
Y = zeros(y_row_count, num_labels);
for i = 1:y_row_count
  Y(i, y(i)) = 1;
end

costOfEachUnit = -Y .* log(A_l3) .- (1.0 .- Y) .* log(1.0 .- A_l3); 
% 5000x10 .* 5000x10 = 5000x10
sumOfCostOfEachUnit = sum(sum(costOfEachUnit, 1),2); 
% sum(5000x10, 1) -> 1x10
adjustment1 = sum(sum((Theta1(:, 2:end) .* Theta1(:, 2:end)), 1), 2);
adjustment2 = sum(sum((Theta2(:, 2:end) .* Theta2(:, 2:end)), 1), 2);

J = (1.0 / m) * sumOfCostOfEachUnit + (lambda/(2*m)) * (adjustment1 + adjustment2);


% --------------------- backpropagation -----------------------

D_2 = zeros(size(Theta2)); % 10x26
D_1 = zeros(size(Theta1)); % 25x401

for i = 1:m
  a_1 = X(i, :); % 1x401
  z_2 = a_1 * Theta1'; % 1x401 * 401x25 = 1x25
  a_2 = 1.0 ./ (1.0 .+ exp(-z_2)); % 1x25

  a_2 = [ones(1,1) a_2]; % 1x26
  z_3 = a_2 * Theta2'; % 1x26 * 26x10 = 1x10
  a_3 = 1.0 ./ (1.0 .+ exp(-z_3)); % 1x10

  d_3 = a_3 .- Y(i, :); % 1x10 .- 1x10 = 1x10
  d_2 = (d_3 * Theta2) .* (a_2 .* (1 .- a_2)); % 1x10 * 10x26 = 1x26 .* 1x26 = 1x26

  D_2 = D_2 .+ (d_3' * a_2); % 1x10' * 1x26 = 10x26
  D_1 = D_1 .+ (d_2(:, 2:end)' * a_1); % 1x25' * 1x401 = 25x401

end

Theta2_grad = ((1 / m) .* D_2) .+ ((lambda/m) .* Theta2); % 10x26 .+ 10x26
Theta2_grad(:, 1) = (1 / m) .* D_2(:, 1); 

Theta1_grad = ((1 / m) .* D_1) .+ ((lambda/m) .* Theta1); % 25x401 .+ 25x401
Theta1_grad(:, 1) = (1 / m) .* D_1(:, 1); 


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
