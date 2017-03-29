function J = costFunctionJ(X, y, theta)

% X is the design matrix containing our training examples
% y is the class labels

m = size(X, 1);
predictions = X * theta;            % y = xのグラフについての仮説関数h_thetaなので、単純に　x * theta でおけ
sqrErrors = (predictions - y) .^ 2; % squared errors. 二乗誤差

J = 1 / (2*m) * sum(sqrErrors);
