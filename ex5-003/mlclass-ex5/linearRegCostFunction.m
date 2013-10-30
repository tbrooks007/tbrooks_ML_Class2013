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

h = (X * theta - y);

% calulate the regularized term (do not include theta 0)
regularized_term = sum(theta(2:size(theta)).^2) * (lambda / (2 * m)); 

% calcuate the cost of theta params with regularziation term
J = (sum((h).^2) / (2 * m)) + regularized_term;

% minimize our parameter values (theta)

% calculate regularized term for gradient (do not include theta 0)
regularized_term_gradient =  theta(2:size(theta)) * (lambda / m);

%calculate gradient
gradient_first_term = (X' * (h)) / m;
grad = gradient_first_term  + vertcat(0, regularized_term_gradient) ; %concat 0 as the first row...for theta 0

% =========================================================================

grad = grad(:);

end
