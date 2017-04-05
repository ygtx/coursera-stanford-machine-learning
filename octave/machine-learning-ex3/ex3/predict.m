function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
A2 = X * Theta1';

ones_A2 = ones(size(A2, 1), 1);
A2h = ones_A2 ./ (ones_A2 .+ (e .^ -A2));

A2h = [ones(size(A2, 1), 1) A2h];
A3 = A2h * Theta2';

ones_A3 = ones(size(A3, 1), 1);
A3h = ones_A3 ./ (ones_A3 .+ (e .^ -A3) );

for i = 1:size(A3h, 1)
  row = A3h(i, :);
  [W, IW] = max(row, [], 2);
  p(i, 1) = IW;
end






% =========================================================================


end
