---
author: chrisjmccormick
comments: true
date: 2013-02-27 23:15:16 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/02/27/canny-edge-detector/
slug: canny-edge-detector
title: Canny Edge Detector
wordpress_id: 5458
---

The following are my notes on part of the Edge Detection lecture by Dr. Shah: **[Lecture 03 – Edge Detection](http://www.youtube.com/watch?v=lC-IrZsdTrw)**

****Canny Edge Detector
****



	
  * Critera:

	
    * Good detection: Minimize false positives and false negatives

	
    * Good localization: Should be close to true edge

	
    * Single response constraint: Only one result for each edge point.




	
  * Uses gradient of Gaussian.


****Steps:
****



	
  1. Smooth image with Gaussian filter

	
  2. Compute derivative of smoothed image.

	
  3. Find gradient magnitudes and orientations.

	
  4. Apply non-maximum suppression

	
  5. Apply hysteresis threshold


****Gaussian Smoothing and Derivative
****



	
  * The first two steps are very similar to the Prewitt operator, where we end up with two images, one is the derivative with respect to x, and the other is the derivative with respect to y.


****![](https://lh4.googleusercontent.com/sQ33npRYYBlzAKVwW3d3srhkIo_3g5CfWLlPkLvC9WDCVWf1I1gjFkN8HpeQkcLEcs9oc6x74UHf1R7tXuBLRamRIeq1jibwGfHCM_RsODv7xdKiKeOajFSL)
[28:11 Lecture 3]
****



	
  * The difference is simply that with the Prewitt operator we used simple averaging for smoothing, but here we’re using the Gaussian for smoothing.


****Non-maximum Suppression
****



	
  * One of the goals is to only have one result point per edge pixel, so in the neighborhood of an edge pixel, we want to find the pixel which is most likely the edge, and suppress the others.

	
  * Look along the normal to the edge. There should only be one edge pixel on the line normal to the edge, so take the one with the largest magnitude.


****
![](https://lh6.googleusercontent.com/331S0kKaTb8BOu6cZlCWtGDd0dJxMeVAxiGBy7rT1-Kyfi9H19QyXx_p5mHuvDgh69_unhR81FHVg9MCHzvYzR7DzE6vELYfDCv7Vvj-uMzmev0zZUhDb3Xd)
[31:37 - Lecture 3]

****



	
  * If a result point is larger than the edge point to its left and to its right along the normal, then it is a maxima and we keep it. Otherwise, ommitt it.


****![](https://lh5.googleusercontent.com/hk6vMKjjxeE7YX2HlRWPVpsN9-WE53WYCFrJg4xy4YLqcUsl55UlKpKhGpQ4xiAvJgZIVfXKAmjTZnbX_pDIt3DuMHIXrNFIzcOQfMsap1eDOTfVEO2oS5yz)
[32:43 Lecture 3]
****



	
  * The first image shows the gradient magnitude, the second image excludes the edge points which are not local maxima, and the third image applies a threshold to identify the edge pixels.


****
Hysteresis threshold
****



	
  * Set a “high” threshold and a “low” thresholds.

	
    * Pixels outside of these thresholds are immediately categorized.

	
    * For pixels between the threshold:

	
      * Check all of it’s neighbors. If any of it’s neighbors are either certain or possible edge pixels, keep it. If not, ommit it.









	
  * The actual algorithm, as its programmed:

	
    * Work through the image left to right then top to bottom.

	
    * If a pixel is above the high threshold, mark it as an edge.

	
      * Then, check all of its neighbors. If they are above the low threshold, then mark them as an edge (since they are touching a real edge)








