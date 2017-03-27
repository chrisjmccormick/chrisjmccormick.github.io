% ======== runRBFNFuncApproxExample2 ======== 
% Trains an RBFN for function approximation on an example dataset with
% 10 dimensions.

% $Author: ChrisMcCormick $    $Date: 2017/03/27 22:00:00 $    $Revision: 1.0 $

addpath('kMeans');
addpath('RBFN');

% =================================
%             Dataset
% =================================

% Load in the dataset.
X = load('func_approx_dataset.csv');

% The last column contains the output values.
y = X(:, size(X, 2));

% Remove the last column.
X = X(:, 1:(size(X, 2) - 1));

fprintf('Datset contains %d points with %d dimensions.\n', size(X, 1), size(X, 2))

% =================================
%       RBFN Properties
% =================================

% 1. Specify the number of RBF neurons.
numRBFNeurons = 10;

% 2. Specify whether to normalize the RBF neuron activations.
normalize = true;

% 3. Calculate the beta value to use for all neurons.
    
% Set the sigmas to a fixed value. Smaller values will fit the data
% points more tightly, while larger values will create a smoother result.
sigma = 10;

% Compute the beta value from sigma.
beta = 1 ./ (2 .* sigma.^2);

% ==================================
%            Train RBFN
% ==================================

fprintf('\nTraining an RBFN on the data...\n');

% Train the RBFN for function approximation.
[Centers, betas, Theta] = trainFuncApproxRBFN(X, y, numRBFNeurons, normalize, beta, true);

% =================================
%        Evaluate RBFN
% =================================

% Evaluate the trained RBFN over the training points.

fprintf('\nEvaluating trained RBFN over the training data...\n');

% Create a vector to hold the 'predicted' values (the output of the RBFN).
p = zeros(length(y), 1);

% For each training sample...
for (i = 1:size(X, 1))

	% Evaluate the RBFN at the query point xs(i) and store the result in ys(i).
	p(i) = evaluateFuncApproxRBFN(Centers, betas, Theta, true, X(i, :));
    
end

% Calculate the average error over the training data.
avg_error = mean(abs(y - p));

fprintf('  Average error over training set: %.3f\n', avg_error);
