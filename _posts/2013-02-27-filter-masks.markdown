---
author: chrisjmccormick
comments: true
date: 2013-02-27 22:45:08 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/02/27/filter-masks/
slug: filter-masks
title: Filter Masks
wordpress_id: 5449
---

Filter masks are fundamental to the implementation of image filters, which are used in many computer vision algorithms.

A filter mask is just a way of representing an operation to be performed on each pixel of an image which incorporates the values of neighboring pixels. It can be represented as a matrix, so it gives us a way to write mathematical equations representing the operation.

# Reference
	
  1. University of Central Florida (UCF) Lecture on YouTube: [Lecture 02 – Filtering](http://www.youtube.com/watch?v=1THuCOKNn6U)
    * The discussion of Convolution and Correlation run from 17:45 to 22:00 in the video.
  2. University of  Nevada, Reno (UNR) Lecture Slides: [Image Processing Fundamentals](http://www.cse.unr.edu/~bebis/CS474/Lectures/SpatialFiltering.ppt)	
  3. [ImageMagick](http://www.imagemagick.org/Usage/convolve/) documentation on Convolution.


## Correlation

The discussion of filter masks starts at about 17:45 in Lecture 2 from Dr. Shah. He makes a jump here from the concept of the derivative mask to correlation which can be confusing. Correlation is a broader concept than just image derivatives, it can be used to apply other kinds of filters as well.

We are working towards the broader concept of image filters. With an image filter, you apply a transformation to an image in which the value of each pixel is changed by considering the values of its neighbors.

He uses the below equation to express this mathematically, where _f_ is the image and _h_ is the mask to be applied. However, _f_ and _h_ are reversed on the right hand side of this equation.

![Image filter equation](https://lh6.googleusercontent.com/wrvyBOSBimOBze7SFc4ZZojtNqNwn7ewrVuNBVKXwGf6qGQHyM4FQ23wk5WA0HIE6QzORIAWcUWwKFhWVXEURx5Tvyk6x9YVgkD5rjsSCChW7FohNkkoSaCi)

I prefer this version of the equation from the 'Image Processing Fundamentals' slides:

![Image filter equation 2](https://lh3.googleusercontent.com/QWZKIBqjuMWcY_MB1x0M-4zb_ec1EVUo30KDSL8TQ8mgirz32sZuGYJ22S1KPh8Ot1Cw0uBza_qoSXpsvSJLzQbt6dvh_b6QxNTlIJkZ_SNrnHroP-F9rKZM)

To compute the output value for the pixel at x, y:  For each position in the mask (for each column _s_ of the mask, and each row _t_ of the mask), multiply the mask value w(s, t) times the corresponding neighbor of pixel x, y in the original image (the neighbor at x + s, y + t.) Take the sum of all of those products, and that is the pixel value at x, y in the result image.

The symbol between f and h denotes correlation.

![](https://lh3.googleusercontent.com/32XmgWiTtYGKddpefnBXs9tFSyfIvDE31TN6hs_PYrWb_lAQGpM6oOF7cvIL9jRmlt2cJoMOTS0drDEhEPnwIkCwC7F4dVYiedVk9nYxdnl-t8AvaIaNViIi)
`[19:26]`

The mask is also referred to as a Kernel.


## Convolution 

The convolution is similar, except that you flip the mask matrix vertically then horizontally before applying it.

![](https://lh6.googleusercontent.com/NmvsgSVGR1IB6UAg4wA1YYACYfOl53OeDVlp4Ol_vyXbEyfvEylFnyZ4eZ8cfbZdBVZ_SnxEnK32r-cS6UZBfJa16c6KBsPxzT4LXxtJM93vD2PfE9m1sfYg)

Neither the video lecture nor the slides explicitly explain the _practical_ importance of the distinction between convolution and correlation.  What's more, all of the applications of image masks that I have seen (Gaussian filter, and Laplacian of Gaussian filter), it is always convolution which is used, even though the masks are symmetrical and rotating them has no effect!

Fortunately, I found [this site](http://www.imagemagick.org/Usage/convolve/#convolve_vs_correlate) which does explain the distinction. The distinction is only important when you have an asymmetrical mask, as you might find, for example, in a Convolutional Neural Network. When the mask is asymmetrical, then only the convolution has the Commutative and Associative Properties that you expect from multiplication--the correlation does not.