---
author: chrisjmccormick
comments: true
date: 2013-03-18 20:08:30+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5469
published: false
slug: sift-object-recognition-tutorial-part-ii-keypoint-detection
title: SIFT Object Recognition Tutorial - Part II - Keypoint Detection
wordpress_id: 5469
---

**Sources**

[AI Shack article](http://www.aishack.in/2010/05/sift-scale-invariant-feature-transform/)


This is a well written article, and I found it to be better illustrated than Dr. Shah's lecture below. He doesn't go into as much detail with the math, but the main points are very clear. This article covers keypoint detection and feature extraction, but stops there.


UCF Computer Vision Course,** [Lecture 5 - SIFT](https://docs.google.com/document/d/141XpZooA528l3tNLHqfYIl2X8dblu6F8ZsPl6oMhfs4/edit#heading=h.pvi1urw7bx76)**


I found Dr. Shah's lecture on SIFT difficult to follow, but it did provide some additional insights that were very valuable. If you start with the AI Shack article, I think you'll be able to get more out of Dr. Shah's lecture.


COSM Computer Vision Course, [EGGN 512 Lecture 12-1 SIFT](http://www.youtube.com/watch?v=U0wqePj4Mx0&list=PL4B3F8D4A5CAD8DA3&index=27)

**Notes on UCF Lecture**

SIFT is a big deal, one of the most influential papers, published in 2004, with over 15,000 citations. David Lowe at University of British Columbia, SIFT is patented by UBC.****

Rotation invariance:



	
  * Harris corner detector is actually rotation invariant: If you take an image and rotate it around its z-axis (normal image rotation), you will still find corners in the same locations of the image.

	
  * SIFT is “not directly” rotation invariant.


****
****Simple transformations:



	
  * Scale

	
  * Rotation

	
  * Translation


****
****Then, there’s sheer, “and so on” :) - affine distortions

SIFT, in a way, addresses the problem of 3D viewpoint. If you take an image of the same object from a different viewpoint, it changes the image significantly.********

SIFT is robust to noise and changes in illumination.

SIFT Advantages:****
****



	
  * Locality: Features are local and therefore robust to occlusion and clutter.

	
    * He explains that a local feature is one which considers a small neighborhood of pixels (like the 5x5 or 10x10 window in the Harris detector).

	
    * From my experience, this is because an object is described by multiple features, and you don’t need to recognize every single one to be confident you’ve found the object.




	
  * Distinctiveness: “Individual features can be matched to a large database of objects” - This is saying that the SIFT keypoints representing an object are unlikely to occur on a different object?

	
  * Quantity: Many interest points even for a small object

	
  * Efficiency: Close to real time performance![](https://lh4.googleusercontent.com/qtp9_el7uWtacYSOABv0zvGr-Wn2bmZ_ifdEi_sBkuf1bqxjU7Nu5WYPpfVj1m5umTM_0MBmP6dP5vwpRiBfVi3gBE4PX8s2r8wD8ERxAodIMvJy3JGUrOC4)


****
[5:35 Lecture 05]****





	
  * Locating the keypoints is an important step, but its the actual descriptor that you get which is of real value.

	
    * Harris gave you the interest points, but not a descriptor.




	
  * According to Dr. Shah, “many people use a Harris detector, but then a SIFT descriptor”.![](https://lh4.googleusercontent.com/_Whp74rOb1JrOWDCQ6MoUBqr4at6x7KmpkpUUZSInM112C_TPDPHZS5sY6Z4vMJ5UVP4u7Jl0XNDHmUNOIrnz0jn1N-rtGrfZMk6cYIcWfPzgWQtE3xzT9da)




[26:00 Lecture 5]



	
  * The SIFT keypoint detector looks for pixels whose laplacian of gaussian (2nd derivative of smoothed image) is either a local minima or local maxima in a 3x3x3 cube, where the the third dimension is the next and previous scales. (The illustration for this point was earlier on in the lecture).

	
  * The laplacian of gaussian can be approximated by subtracting the gaussian filtered image at one scale from the gaussian filtered image at the next scale (see the diagram above)

	
    * Each subsequent scale uses a sigma value which is sqrt(2) times larger than the value for the previous scale.




	
  * Note how the next octave has one fourth the resolution of the previous octave.![](https://lh6.googleusercontent.com/-MEC_E9gnxh5MP33OqOZv-3nkUExjagzLsRvoLbUodlTcKT5pQAbY6vB6ZDU6md-msiWP4Lk2qGwDdYGu_qFerqkERjaVA4zcm7H9i35Y3kq2_U9m-1Vk-Jr)


****
[29:08 Lecture 5]
****The above diagram shows the value of sigma used for each scale and each octave.****
****



	
  * As the scale increases (from left to right in the table) we multiply sigma by sqrt(2).

	
  * Note how we start with a larger initial sigma at each octave (the largest image size, which is the first octave, is the first row of the table)![](https://lh4.googleusercontent.com/fZGQoDWzwcKkimnShLdDAWGjkRWLZV-Y-WTxaAl-98jYIVOxgl6n-QdiE6RGLYfC62LFFPFOAwVcsm7iMZ9SJ1XrN-ucVZffNAWPuAU8GjD10u3H7kMW5_F6)


****
[36:17 Lecture 5]
****



	
  * He points out the difference between what we’re looking for here versus the Marr-Hildreth edge detector

	
    * The Marr-Hildreth edge detector used the LoG to find edges by looking for zero crossings.

	
    * Here we’re looking for interest points which we’re defining as maxima or minima in the LoG across multiple scales.





****
****Filtering out bad keypoints:****
****



	
  * Remove keypoints where the LoG value is small (below a threshold). The aishack post refers to these as “low contrast” interest points.

	
  * Only keep keypoints that correspond to corners.


****
****Compute the orientation of the keypoint.****
****



	
  * Each keypoint exists within a specific scale. Compute the gradient vector at the keypoint location in its scale image.


****
****Using gradients for feature vector over intensity values:****
****



	
  * If illumination changes, the intensity values in the ROI will change a lot

	
  * However, the gradient vectors will stay more constant.


****
****Use KNN to find the closest matching features****
****



	
  * If the ratio between 1st and 2nd match distance is less than 0.8, then disregard this match altogether. Better to toss the match then mistakenly match to wrong feature.


