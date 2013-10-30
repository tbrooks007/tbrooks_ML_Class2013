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

% add ones column to the input (X) matrix
X = [ones(m, 1) X];

% caculate the activation unit outputs for hidden layer a2
hidden_layer_a2 = sigmoid(X * Theta1');

% add ones column to the output (matrix) of activation units of layer 2 (a2)
size_a2 = size(hidden_layer_a2, 1)
hidden_layer_a2 = [ones(size_a2, 1) hidden_layer_a2];

% propagate the hidden layer outputs to the output layer (a3)...calculate the hypthosis...get our predictions
g = sigmoid(hidden_layer_a2 * Theta2');

% get the max values for each row as well as the index of the max value...add the max index value to p
% as that is the value that equales the correct class we are prediciting for values of x (example rows) to be
[b,dx] = max(g, [], 2);
p = dx;



% =========================================================================


end
