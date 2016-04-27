---
author: chrisjmccormick
comments: true
date: 2013-02-27 22:58:24 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/02/27/gaussian-filter/
slug: gaussian-filter
title: Gaussian Filter
wordpress_id: 5452
---

**Reference**



	
  1. University of Central Florida (UCF) Lecture on YouTube:** ** **[Lecture 02 – Filtering](http://www.youtube.com/watch?v=1THuCOKNn6U)**

	
    * The discussion of the Gaussian filter runs from 22:40 to 35:44 in the video.




	
  2. University of Central Florida (UCF) Lecture on YouTube: **[Lecture 03 – Edge Detection](http://www.youtube.com/watch?v=lC-IrZsdTrw)**


****Gaussian
****



	
  * A gaussian mask falls off exponentially as you move away from the center pixel.

	
  * You can smooth an image by replacing each pixel with the average of its neighbors, giving each neighbor equal weight (the mask or “kernel matrix” would be all 1's). This creates a blurry image.

	
  * If you instead do a weighted average where the weights fall off exponentially in a gaussian distribution, you can get much nicer smoothing.

	
  * Averaging and Gaussian smooting are given as examples of removing noise.

	
  * A larger sigma value will increase the smoothness.

	
  * In order to get a full gaussian curve in your mask, you need to have a large enough mask size. 3x3 is not big enough.

	
    * Larger standard deviations (sigma) require a larger mask size.

	
    * For example, with sigma = 1, you need at least a 7x7 mask





****![](https://lh3.googleusercontent.com/t1Yyuw23t0VXraEDTMOIwBbcZv7J_HmL0miNFxlf3YmRVWDa_uvLDkYqpjN5blssp3Dz16KecjU1ZjUC-PBAJA4s7oo8O_-02HsxjQotWD6rvGJfpvP2hbQg)
****


[6:45 in Lecture 3]


The below example shows the 2D Gaussian, with each mask position multiplied by 255. Using integral values is better so that you don’t have to use floating point values.

****![](https://lh5.googleusercontent.com/QLcuKxalija9VG0R-xtVjbo5WGbBCwP9-O2tHxsIFIGh_v4JCUHHPt88vdYZaGV3aXp_htUMxKcFg-OhLZRizL2KT-f76QIaiL_SpzlYS2ObJVjvbLXAwK_B)
****

[7:26 in Lecture 3]


The Gaussian filter is said to be “separable” [12:40 Lecture 3], and this has very important performance implications. This concept had me tripped up for a while.




Normally, when applying a 2D mask, you visit every pixel in the image once and apply the mask to it, requiring n-squared (where n is the mask size) multiplications at each pixel location in the image.




Because the Gaussian filter is separable, you can take the image in two passes. In the first, you take a vertical 1-D mask and apply it to every pixel in the image. Then, you take the resulting image and apply a horizontal 1-D mask to every pixel in the image. The result is equivalent to the single pass with the 2D mask, but only requires 2n multiplications per pixel instead of n-squared.




This had me confused initially because I thought we were applying a 1D vertical mask and a 1D horizontal mask in a single pass of the image, so that the resulting value at pixel 'a' would only incorporate the values of the pixels in the same row as 'a' plus the pixels in the same column as 'a', omitting all of the other pixels in the mask area.




This is not the case because the 1D masks are applied in two passes, first the vertical mask, then the horizontal mask. If you think through what this looks like, you’ll see that every pixel in the mask area ends up contributing to the final pixel value.
