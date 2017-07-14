function [ output_args ] = getHyperParamError( X, y, numRBFNeurons, beta )
%GETHYPERPARAMERROR Summary of this function goes here
%   Detailed explanation goes here

    % ==================================
    %            Train RBFN
    % ==================================

    % Train the RBFN for function approximation.
    [Centers, betas, Theta] = trainFuncApproxRBFN(X, y, numRBFNeurons, normalize, beta, true);

    % =================================
    %        Evaluate RBFN
    % =================================

    % Evaluate the trained RBFN over the training points.

    % Create a vector to hold the 'predicted' values (the output of the RBFN).
    p = zeros(length(y), 1);

    % For each training sample...
    for (i = 1:size(X, 1))

        % Evaluate the RBFN at the query point xs(i) and store the result in ys(i).
        p(i) = evaluateFuncApproxRBFN(Centers, betas, Theta, true, X(i, :));

    end   

    % Calculate the Mean Squared Error over the training data.
    mse = mean((y - p).^2);
    
    return(mse);
end

