function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


m = length(X);

%iterate over all examples
for example_idx = 1:m

	sqrd_distance = zeros(K,1);
	x = X(example_idx,:); %get current example

	%iterate over all centroids, and calcuate the distance between it and the current example
	for j = 1:K
		centroid = centroids(j,:);
		diff = (x - centroid);
		sqrd_distance(j) = diff * diff'; %calculate squared distance
	end

	%get minimize distance
	[value, idx_min]= min(sqrd_distance);
	idx(example_idx) = idx_min;
end




% =============================================================

end

