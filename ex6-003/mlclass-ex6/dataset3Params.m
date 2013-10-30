function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possible_sigma_c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
len = length(possible_sigma_c_values);
prediction_errors = [length(possible_sigma_c_values), length(possible_sigma_c_values)];


%iterate over all possible combinations of C and sigma
for penality_i = 1:len
	for sigma_i = 1:len

		%get values for C and sigma
		C = possible_sigma_c_values(penality_i)
		sigma =  possible_sigma_c_values(sigma_i)

		%train classifiers with current values of C and sigma
		model = svmTrain(X, y, C, @(x1,x2) gaussianKernel(x1, x2, sigma));

		%predict values based on cv data set
        predictions = svmPredict(model, Xval);

        %compute prediction error
        prediction_errors(penality_i, sigma_i) = mean(double(predictions ~= yval));
	end
end

% find the minium values for C and sigma
minium_matrix = min(min(prediction_errors));
[c_minimum, sigma_minimum] = find(prediction_errors == minium_matrix);

C = possible_sigma_c_values(c_minimum);
sigma = possible_sigma_c_values(sigma_minimum);

% =========================================================================

end
