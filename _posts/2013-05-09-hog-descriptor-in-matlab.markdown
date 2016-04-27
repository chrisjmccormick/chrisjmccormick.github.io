---
author: chrisjmccormick
comments: true
date: 2013-05-09 05:13:42 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/05/09/hog-descriptor-in-matlab/
slug: hog-descriptor-in-matlab
title: HOG Descriptor in MATLAB
wordpress_id: 5637
tags:
- Feature Extraction
- Histograms of Oriented Gradients
- HOG
- HOG Descriptor
- MATLAB
- Octave
- Person Detection
- Person Detector
---

To help in my understanding of the HOG descriptor, as well as to allow me to easily test out modifications to the descriptor, I wrote functions in Octave for computing the HOG descriptor for a detection window.

**HOG Tutorial**

For a tutorial on the HOG descriptor, check out my [HOG tutorial post](http://chrisjmccormick.wordpress.com/2013/05/09/hog-person-detector-tutorial/).

**Update: Revision 1.2**



	
  * Thanks to Carlos Sampedro Pérez for identifying the bellow issues.

	
  * Fixed a bug in getHOGDescriptor--it was referencing 'getUnsHistogram' instead of 'getHistogram'

	
  * The 'rows' and 'columns' calls were causing problems for Matlab users, so I replaced these with 'size' calls. Also expanded '+=' operators for Matlab users.


**Source files**

[getHOGDescriptor.m](https://dl.dropboxusercontent.com/u/94180423/getHOGDescriptor.m) - Computes the HOG descriptor for a 66x130 pixel image / detection window. The detection window is actually 64x128 pixels, but an extra pixel is required on all sides for computing the gradients.

[getHistogram.m](https://dl.dropboxusercontent.com/u/94180423/getHistogram.m) - Computes a single 9-bin histogram for a cell. Used by 'getHOGDescriptor'.

Octave code is compatible with MATLAB, so you should also be able to run these functions in MATLAB.

**Differences with OpenCV implementation**



	
  * OpenCV uses L2 hysteresis for the block normalization.

	
  * OpenCV weights each pixel in a block with a gaussian distribution before normalizing the block.

	
  * The sequence of values produced by OpenCV does not match the order of the values produced by this code.


**Order of values**

You may not need to understand the order of bytes in the final vector in order to work with it, but if you're curious, here's a description.

The values in the final vector are grouped according to their block. A block consists of 36 values: 1 block  *  4 cells / block  * 1 histogram / cell * 9 values / histogram = 36 values / block.

The first 36 values in the vector come from the block in the top left corner of the detection window, and the last 36 values in the vector come from the block in the bottom right.

Before unwinding the values to a vector, each block is represented as a 3D dimensional matrix, 2x2x9, corresponding to the four cells in a block with their histogram values in the third dimension. To unwind this matrix into a vector, I use the colon operator ':', e.g., A(:).  You can reshape the values into a 3D matrix using the 'reshape' command. For example:


<blockquote>% Get the top left block from the descriptor.

block1 = H(1:36);

% Reshape the values into a 2x2x9 matrix B1.

B1 = reshape(block1, 2, 2, 9);</blockquote>


**Past Versions**

**v1.1**

Revision 1.1 includes some important changes:



	
  * The getHOGDescriptor function now requires that the input image be 130 pixels tall by 66 pixels wide. This is to provide a 1-pixel border for properly computing the gradients at the edges of the 128x64 detection window.

	
  * The original getHistogram function took a 'numBins' parameter was still partially hardcoded to 9-bins. I've fixed this.

	
  * The getHistogram implementation now computes the _unsigned_ histogram, which gives better results for person detection.

	
  * I've flipped the 'hy' filter mask to match the mask used by OpenCV.


Note that the descriptors computed by v1.1 of these files are not compatible with descriptors computed by the v1.0 files, so they shouldn't be intermixed.

[getHistogram_v1.1.m](https://dl.dropboxusercontent.com/u/94180423/getHistogram_v1.1.m)

[getHOGDescriptor_v1.1.m](https://dl.dropboxusercontent.com/u/94180423/getHOGDescriptor_v1.1.m)

**v1.0**

The original release.

[getHistogram_v1.0.m](https://dl.dropboxusercontent.com/u/94180423/getHistogram_v1.0.m)

[getHOGDescriptor_v1.0.m](https://dl.dropboxusercontent.com/u/94180423/getHOGDescriptor_v1.0.m)

**Send your feedback**

Please let me know if you find any bugs, opportunities for optimization, or any other discrepancies from the original descriptor.
