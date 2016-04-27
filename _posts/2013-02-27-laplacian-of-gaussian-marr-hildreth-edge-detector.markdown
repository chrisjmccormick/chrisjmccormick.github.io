---
author: chrisjmccormick
comments: true
date: 2013-02-27 23:10:55 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/02/27/laplacian-of-gaussian-marr-hildreth-edge-detector/
slug: laplacian-of-gaussian-marr-hildreth-edge-detector
title: Laplacian Of Gaussian (Marr-Hildreth) Edge Detector
wordpress_id: 5455
---

The following are my notes on part of the Edge Detection lecture by Dr. Shah: **[Lecture 03 – Edge Detection](http://www.youtube.com/watch?v=lC-IrZsdTrw)**



	
  * Noise can really affect edge detection, because noise can cause one pixel to look very different from its neighbors.

	
  * To account for this, we take advantage of the fact that neighboring pixels along an edge tend to look similar.

	
  * Gaussian smoothing helps eliminate noise. The larger the sigma, the greater the smoothing.

	
  * The simplest edge detectors are the Prewit and Sobel edge detectors. These are pretty old.

	
  * Laplacian of Gaussian (Marr-Hildreth) is better.

	
  * Even better: Gradient of Gaussian (Canny)


****Prewitt and Sobel:
****



	
  1. Compute derivatives in x and y directions.

	
  2. Find gradient magnitude

	
  3. Threshold the magnitude.


****
Copmuting derivatives:
****First you smooth out the images by averaging with the mask:
The below image shows the Prewitt mask in the x-direction. The right hand side represents the operator as two operations. The vertical vector represents a smoothing of the image by averaging. The horizontal vector represents the derivative in the x direction. These two operations can be combined into a single mask as shown on the left hand side.
****![](https://lh4.googleusercontent.com/zIQWgTwtVpUjL4cZ0k01BWO7s-DF7sLXu-zGtGdf3RZtb5vNAhLV--8CUAlcUOrfHb1UWv6AMvKjwP7puhHdPuc8LFKmMZ6pB7w2DPNjKvwGyYdauzNHg1qw)

****Copmuting the magnitude of the vector incorporates both the the derivative in the x and y direction, and this ensures that we’re able to detect edges at any angle (not just horizontal or vertical edges).
****
Marr-Hildreth
****The Marr-Hildreth operator is also called the Laplacian of Gaussian, which I saw referenced in SIFT...****
****



	
  1. Apply Gaussian smoothing

	
  2. Take the second derivative and look for zero crossings (where 2nd derivative = 0, but is not constant 0)

	
    1. When the first derivative is at a maxima or minima the second derivative is 0.

	
    2. Pixels where a zero crossing occurs are marked as edges (if the slope of the crossing exceeds a threshold).

	
    3. Look for zero crossings along each row (why not by columns, too?)






	
  * According to Wikipedia, this edge detector is more of historical significance because it has a couple serious flaws, and the Canny edge detector is better.

	
  * Marr-Hildreth performs the smoothing using the Gaussian instead of just simple averaging. We already know that the Gaussian does a better job of smoothing, so that’s already one improvement.

	
  * The second order derivative is the Laplacian?




![](https://lh5.googleusercontent.com/QdrfeTHuZKvVwDXz5dow2rxrqz52LY09oAXEK5SRxeY4py3wnqG_iSzi5MryitBKWzuOWcfIh-Ikulx7lWO5nxQ57FvLubOtM4FZPuiBiRZmHVFtAYwcn-H7)![](https://lh6.googleusercontent.com/Fz6bUYWUjRveTofgVZCq1BkmFwauNL6boifrIYNSiN1BFtKrkmhy-r5LKrfMWW14plHZsmSBUxtydmYVdti4i0VAoOtf_oxoeh0dnC8EDOIbVf4NLznkcdmn)






	
  * Rather than convolve the image with the gaussian and take the second order derivative of the result, you can actually take the 2nd derivative of the gaussian itself.




****![](https://lh4.googleusercontent.com/iJNpWfWO2_PKWNCiPAb5op2eHmC5drEP3rjiZJslISwtLUqboGFgCCpnbtj720L6tpWq1TQ1oOI3NGXXY2u-5W1_3fL7KRkxsV3kxzZALR699oru8SMGWUA-)

****






	
  * You can precompute the mask values, as shown below. He generally uses a sigma of 1, so I assume that’s the case here:




**** ![](https://lh5.googleusercontent.com/aalwn2u86_tJFv8nwkcFA555F3ppcu40PfjahcmxO8Q1LlLTCPp2oFXp1vDpVO3JCvVceVtzQePLcM2t0I0LusDc_PViIUYQgH44h6BBaf6suyLfiQN2HI8q)
****




[8:14 Lecture 3]






	
  * Zero crossings occur in the image wherever a positive value is followed by a negative value, or vice versa. Or there may even be a 0 in between a positive and negative value.

	
    * The absolute difference between the negative and positive values gives you the slope of the crossing, which is a measure of the strength of the edge.






	
  * In the same way that we were able to separate the Gaussian filter to improve performance, we can separate the Laplacian of Gaussian




****![](https://lh6.googleusercontent.com/M1K823S4TxPfQjlMK4bFPeJodm7Ll_Wjhgdp3hTtFrrurDugLRo9QNbUN2dqq9V1tP91kW1Vugk0jCgJnbaSrpC-V_KkoIeruNfDbI4lLC7xZ53WCGBmmY3m)
[15:40 Lecture 3] - Note: g(x) and g(y) should be switched!

****






	
  * The Gaussian filter can be applied with 2n multiplications (where n is the mask size), but the LoG requires 4n.

	
  * This requires four passes of the image to apply the 1D vectors (with a final fifth pass to sum the results?)


