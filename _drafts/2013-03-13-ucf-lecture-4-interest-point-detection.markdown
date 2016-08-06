---
author: chrisjmccormick
comments: true
date: 2013-03-13 17:19:28+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5461
published: false
slug: ucf-lecture-4-interest-point-detection
title: UCF Lecture 4 - Interest Point Detection
wordpress_id: 5461
---

****Image Features
****



	
  1. Detect interest points in the image (typically corners)

	
  2. Extract a feature descriptor for each interest point.

	
  3. Feature matching: Determine correspondence between two images.


****Where is this useful?
****



	
  * Tracking: Locate an object in subsequent frames by looking for its features.

	
  * Stereo calibration: Find correspondence between left and right images (which gives you depth information)

	
  * Recognize objects by their features

	
  * Robot navigation - Maybe recognizing features across frames lets it know how it’s moving relative to the object?

	
  * Image retrieval and indexing: Find me an image similar to this one by comparing features.

	
  * Other examples which I understood less:

	
    * “Point matching for computing disparity”

	
    * “Motion based segmentation”

	
    * “3D object reconstruction”





****[3:15]
****



	
  * We want the interest points to be repeatable, so that the algorithm (ideally) always chooses the same features on the object. If it chooses different interest points every time, then we can’t match them.

	
  * We want the feature descriptors to provide some invariance to “geometric and photometric differences”. We want to be able to match two features even if one is more brightly lit than the other, for example.


****
[6:15]
****



	
  * What is an interest point?

	
    * Expressive texture

	
      * A point at which the edge direction changes suddenly (such as a corner, or the intersection of two lines).








**[9:08] - Harris Corner Detector**

Sum of Square Differences

**![](https://lh5.googleusercontent.com/e17u-2M7VUibHnf8VzCXwqq3tKX5eaLqRG6B71YwhMW9uItEvu_DAr7T4pc9z7jtitpiQdZz0MH-GoS2kqOu_Iq3I71dowtDZFAt2f9jdIWDp0fdOGjyxSVR)
**Dr. Shah demonstrates that minimizing the SSD (which you would do to find similarity between two images) is mathematically equivalent to maximizing the Correlation.

Take two windows from an image, one which is shifted by u, v from the other. For each pixel x, y in the windows, take the squared difference of the corresponding pixels, multiply it by a weight from ‘w’, and sum up all of the results.**![](https://lh5.googleusercontent.com/GmZpjq0EmQeGY9m1dZCiBROAo45PtGymYRwAU2k5CsMJwmTgdKwJT4WrcYdVsxcNCRY0CFelWBvtJvK_C5pVZXD1ciCRx_0PSoPbavvs85LdycEN6Wi5u8z8)
**This gives you a measure of how different the two windows are from each other.****

![](https://lh3.googleusercontent.com/53Z9d9jFiiBDKvEeLzEgaq6kIaP7rJ-ROPuM2vyLInc2sNNqqP8hlT_hm12RjXaVcxvh-Dztb_ipxWbwEhyVmkOdqWt9nQNlVe_e4kykYfw5zRC3qWuUu9oS)
[19:40 Lecture 4]

This diagram shows a window around a potential feature. The graph shows the total image difference for all of the possible combinations of u, v around the window.

Notice how in the first image, the total difference is high for most values of u, v. There is one “nice unique minimum”. This suggests the presence of a corner.

In the second image, it looks like there are a number of u,v values for which the total difference is low (the bottom of the trench), suggesting that this feature is an edge (since shifting the window along the edge doesn’t change the window much).

[20:50 Lecture 4]
Using the Taylor series approximation around x, y, this can be approximated as:

**![](https://lh6.googleusercontent.com/2aAEZlUSBATQqhCUdJMTubdvKZdEBK5Skb1AIi0TcjOR9O364Dtl-bEPkXpbiZmCqZOqJ36CNk5WiaF1hE_0Qn2F1JG5FZyorUhNgDuigZmQ8JO_ash9a4te)
**You can work through the Taylor series, but this is a pretty intuitive replacement. Instead of subtracting the values at two points, we’re taking delta x (called ‘u’ here) times the rate of change in Intensity with respect to x (“Ix”) and adding that to the same for y. It makes sense that this is equal to the difference in I at the two points.

He then shows how this can be refactored to the following equation:
**![](https://lh3.googleusercontent.com/EanrhGRCb0_79qFC1o3VJ0A8GWg5Q2fq_QOVWn17zs6qEfqI2S4QwOmasWHyQ2Gzg8dYi7crAhfcKbCMsxUwOQfWiH2bMGpVFisiUO1E4cMEpxY1956OReB9)
**[26:48 Lecture 4]

Note that ‘M’ is a two-by-two matrix. The summation is done over every pixel within a window, then you have single matrix M for that window.

In the left hand equation, note that when you multiply a matrix times a column vector, the result is a column vector. Then when you multiple a column vector times a row vector, the result is a scalar.

Once you have this matrix M, you could look at every possible value of u and v and see if the resulting graph looks like a corner or not. Instead, we can actually just look at the eigen values of the matrix M and make our determination based on those.

Dr. Shah’s discussion of eigen values and eigen vectors is pretty good, it ends at 31:52.**
![](https://lh4.googleusercontent.com/Xhu4KnT3Ae49HjHfbtfLNKVJfYN9RjSMTtZVCJ9qv9MrLS7qMFZdHTCs8TuREgAh_Q6k1pHGS7Q7P5Kax2DUqP2cd3pOkYhAVpze6Ma03yVpwxkVfI7fHrNp)
**[36:26 - Lecture 4]**
**
