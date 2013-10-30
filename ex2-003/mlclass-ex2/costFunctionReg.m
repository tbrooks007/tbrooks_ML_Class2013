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


% note, we transpose the matrices here so that we can actually do the math other wise the function blows up

%predict values using our model and intial theta parameter values
h = sigmoid(X * theta);

% calulate the regularized term (do not include theta 0)
regularized_term = sum(theta(2:size(theta)).^2) * (lambda / (2 * m)); 

% calcuate the cost of theta params with regularziation term
J = (sum(-y' * log(h) - (1 - y') * log(1 - h)) / m) + regularized_term;

% minimize our parameter values (theta)

% calculate regularized term for gradient (do not include theta 0)
regularized_term_gradient =  theta(2:size(theta)) * (lambda / m);

grad = ((X' * (h - y)) / m) + vertcat(0, regularized_term_gradient) ; %concat 0 as the first row...for theta 0

% =============================================================

end
