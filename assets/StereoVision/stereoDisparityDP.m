% Load the stereo images.
left = imread('left.png');
right = imread('right.png');

% Convert the images from RGB to grayscale by averaging the three color channels.
leftI = mean(left, 3);
rightI = mean(right, 3);

% The disparity range defines how many pixels away from the block's location
% in the first image to search for a matching block in the other image.
disparityRange = 50;

% Define the size of the blocks for block matching.
halfBlockSize = 4;

% =============================================
%           Dynamic Programming
% =============================================

% Initialize the empty disparity map.
Ddynamic = zeros(size(leftI), 'single');

% Get the image dimensions.
[imgHeight, imgWidth] = size(leftI);

% False infinity
finf = 1e3; 

% Initialize a 'disparity cost' matrix.
% All values are initialized to a large value ('false infinity').
% The matrix has one row per image column, and one column for each possible 
% disparity value.
% This matrix is used for a single row of the image, and then re-initialized
% for the next row.
disparityCost = finf * ones(imgWidth, 2 * disparityRange + 1, 'single');

disparityPenalty = 0.5; % Penalty for disparity disagreement between pixels

%hWaitBar = waitbar(0,'Using dynamic programming for smoothing...');
fprintf('Using dynamic programming for smoothing...\n');

% For each row of pixels in the image...
for (m = 1 : imgHeight)

	% Update progress every 10th row.
	if (mod(m, 10) == 0)
		fprintf(' Image row %d / %d (%.0f%%)\n', m, imgHeight, (m / imgHeight) * 100);
	end

	% Re-initialize the disparity cost matrix.
    disparityCost(:) = finf;

	% Set min/max row bounds for image block.
	% e.g., for the first row, minr = 1 and maxr = 4
    minr = max(1, m - halfBlockSize);
    maxr = min(imgHeight, m + halfBlockSize);

    % For column of pixels in the image...
    for (n = 1 : imgWidth)
        
		% Set the min/max column bounds for the block.
		% e.g., for the first column, minc = 1 and maxc = 4
		minc = max(1, n - halfBlockSize);
        maxc = min(imgWidth, n + halfBlockSize);
        
		% Limit the search so that we don't go outside of the image. 
		% 'mind' is the the maximum number of pixels we can search to the left.
		% 'maxd' is the maximum number of pixels we can search to the right.
		% Examples:
		%  First column:  mind = 0,   maxd = 15
		%  Middle column: mind = -15, maxd = 15
		%  Last column:   mind = -15, maxd = 0
        mind = max(-disparityRange, 1 - minc);
        maxd = min( disparityRange, imgWidth - maxc);
		
        % Compute and save all matching costs.
		% Compute the SAD between the template at pixel (n, m) and all of the
		% blocks within the search range.
        for (d = mind : maxd)
            %{
			% Right image disparity.
			disparityCost(n, d + disparityRange + 1) = ...
                sum(sum(abs(leftI(minr:maxr,(minc:maxc)+d) ...
                - rightI(minr:maxr,minc:maxc))));
			%}
			% Left image disparity.
			disparityCost(n, d + disparityRange + 1) = ...
                sum(sum(abs(rightI(minr:maxr,(minc:maxc)+d) ...
                - leftI(minr:maxr,minc:maxc))));			
        end
    end

    % Process scan line disparity costs with dynamic programming.
    
	% optimalIndeces will be a lookup table which will tell you what the 
	% disparity should be for the pixel in column k+1 given pixel k's 
	% disparity.
	% optimalIndeces will have 'imgWidth' rows and 31 columns.
	% 
	optimalIndices = zeros(size(disparityCost), 'single');
    
	% Start with the SAD values for the rightmost pixel on the current
	% line of the image.
	cp = disparityCost(end, :);
	
	% For each pixel in the scan line from right to left...
	% (j is initialized to the second to last image column, then iterates
	% towards the leftmost column.)
    for (j = imgWidth-1:-1:1)
        
		% MW - "False infinity for this level"
		% (imgWidth - j + 1) = the number of pixels over we are from the right
		% edge of the image.
        cfinf = (imgWidth - j + 1) * finf;
		
        % Construct matrix for finding optimal move for each column
        % individually.
		% This matrix has:
		%    - 29 columns (2 fewer than the number of disparities)
		%    - The SAD values appear in each row, but shifted 1-pixel
		%      to the left each time as we move down the matrix.
		%  cp is 31 values across.
		% 
		% Find the minimum value in each column of this matrix.
		%     v - becomes a row vector containing the minimum values.
		%    ix - becomes a row vector containing the row index of the min for
		%         each column.
        [v,ix] = min([cfinf cfinf cp(1:end-4)+3*disparityPenalty;
                      cfinf cp(1:end-3)+2*disparityPenalty;
                      cp(1:end-2)+disparityPenalty;
                      cp(2:end-1);
                      cp(3:end)+disparityPenalty;
                      cp(4:end)+2*disparityPenalty cfinf;
                      cp(5:end)+3*disparityPenalty cfinf cfinf],[],1);
        
		% Select the SAD values for the next pixel to the left, and make the
		% following modifications:
		%   - Replace the leftmost and rightmost block SAD value with 'cfinf'
		%     (which grows linearly in magnitude as we move left).
		%   - Add the minimum values from the above matrix to all of the SAD
		%     values.
		cp = [cfinf disparityCost(j,2:end-1)+v cfinf];
        
		% Record optimal routes.
		%                                 Ranges from 2-30            the 
        optimalIndices(j, 2:end-1) = (2:size(disparityCost,2)-1) + (ix - 4);
    end
	
    % Recover optimal route.
	
	% Get the minimum cost for the leftmost pixel and store it in Ddynamic.
    [~,ix] = min(cp);
    Ddynamic(m,1) = ix;
    
	% For each of the remaining pixels in this row...
	for (k = 1:(imgWidth-1))
        % Set the next pixel's disparity.
		% Lookup the disparity for the next pixel by indexing into the 
		% 'optimalIndeces' table using the current pixel's disparity.
		Ddynamic(m,k+1) = optimalIndices(k, ...
            max(1, min(size(optimalIndices,2), round(Ddynamic(m,k)) ) ) );
    end
	
    %waitbar(m/imgHeight, hWaitBar);
end

%close(hWaitBar);

Ddynamic = Ddynamic - disparityRange - 1;

figure(3), clf;
%imshow(Ddynamic,[]), axis image, colormap('jet'), colorbar;
imagesc(Ddynamic), axis image, colormap('jet'), colorbar;
%caxis([0 disparityRange]);
title('Block matching with dynamic programming (Block size = 9)');