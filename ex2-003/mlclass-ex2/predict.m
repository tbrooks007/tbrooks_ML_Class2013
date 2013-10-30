function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

%caculate z for the sigmoid function (these are the same size so no need to transpose the theta matrix)
z = X * theta; 

%caculate the sigmoid function (prediction model for each value in X)
g = sigmoid(z);

%get the indicies for the values in g that are >= 0.5
results_pos = find(g >= 0.5); 

%for each value that is positive set the prediction vector with the corresponding index to 1 otherwise it will be 0
p(results_pos) = 1;

% =========================================================================


end
