---
author: chrisjmccormick
comments: true
date: 2013-05-07 17:22:44 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/05/07/gradient-vectors/
slug: gradient-vectors
title: Gradient Vectors
wordpress_id: 5616
tags:
- Computer Vision
- Gradient Image
- Gradient Vector
- Image Gradient
- Lighting Invariance
---

Gradient vectors (or "image gradients") are one of the most fundamental concepts in computer vision; many vision algorithms involve computing gradient vectors for each pixel in an image.

After a quick introduction to how gradient vectors are computed, I'll discuss some of its properties which make it so useful.


### Computing The Gradient Image


A gradient vector can be computed for every pixel an image. It's simply a measure of the change in pixel values along the x-direction and the y-direction around each pixel.

Let's look at a simple example; let's say we want to compute the gradient vector at the pixel highlighted in red below.

[![dxExample](http://chrisjmccormick.files.wordpress.com/2013/05/dxexample.png)](http://chrisjmccormick.files.wordpress.com/2013/05/dxexample.png)



This is a grayscale image, so the pixel values just range from 0 - 255 (0 is black, 255 is white). The pixel values to the left and right of our pixel are marked in the image: 56 and 94. We just take the right value minus the left value and say that the rate of change in the x direction is 38 (94 - 56 = 38).

Note: At this pixel, the pixels from dark to light as we move left to right. If we looked at the same spot on the left side of the penguin's head where the pixels instead change from light to dark, we'd get a negative value for the change. You can compute the gradient by subtracting left from right or right from left, you just have to be consistent across the image.

We can do the same for the pixels above and below to get the change in the y-direction:

[![dxdyExample](http://chrisjmccormick.files.wordpress.com/2013/05/dxdyexample.png)](http://chrisjmccormick.files.wordpress.com/2013/05/dxdyexample.png)



93 - 55 = 38 in the y-direction.

Putting these two values together, we now have our gradient vector.

[![vector](http://chrisjmccormick.files.wordpress.com/2013/05/vector.png)](http://chrisjmccormick.files.wordpress.com/2013/05/vector.png)

We can also use the equations for the magnitude and angle of a vector to compute those values.

[![PolarCoordinates](http://chrisjmccormick.files.wordpress.com/2013/05/polarcoordinates.png)](http://chrisjmccormick.files.wordpress.com/2013/05/polarcoordinates.png)



We can now draw the gradient vector as an arrow on the image. Notice how the direction of the gradient vector is perpendicular to the edge of the penguin's head--this is an important property of gradient vectors.



[![VectorArrow](http://chrisjmccormick.files.wordpress.com/2013/05/vectorarrow.png)](http://chrisjmccormick.files.wordpress.com/2013/05/vectorarrow.png)



Let's see what it looks like to compute the change in the x and y direction at every pixel for the image. Note that the difference in pixel values can range from -255 to 255. This is too many values to store in a byte, so we have to map the values to the range 0 - 255. After performing this mapping, pixels with a large negative change will be black, pixels with a large positive change will be white, and pixels with little or no change will be gray.

[![GradientImage](http://chrisjmccormick.files.wordpress.com/2013/05/gradientimage.png)](http://chrisjmccormick.files.wordpress.com/2013/05/gradientimage.png)




## Gradient Vector Applications


The first and most obvious application of gradient vectors is to edge detection. You can see in the gradient images how large gradient values correspond to strong edges in the image.

The other less obvious application is to feature extraction. Look at what happens to the gradient vector when I increase the brightness of the image by adding 50 to all of the pixel values.

[![Brightness50](http://chrisjmccormick.files.wordpress.com/2013/05/brightness50.png)](http://chrisjmccormick.files.wordpress.com/2013/05/brightness50.png)



In this brighter image, the rate of change in the x-direction is still 144 - 106 = 38, and the rate of change in the y-direction is still 143 - 105 = 38, the same as in our original image. So even though the pixel values are all completely different, we still get the same gradient vector at this pixel!

When we base our feature descriptors on gradient vectors instead of just the raw pixel values, we gain some "lighting invariance". We'll  compute the same descriptor (or at least closer to the same descriptor) for an object under different lighting conditions, making it easier to recognize the object despite changes in lighting.


## Mathematics


This is a brief introduction to gradient vectors without much use of the mathematical terms or expressions for what we're doing. In another post, titled [Image Derivatives](http://chrisjmccormick.wordpress.com/2013/02/26/image-derivative/), I approach the same topic from a more mathematical perspective. The Image Derivatives post is actually my notes on a computer vision lecture given by Professor Shah which is freely available online.
