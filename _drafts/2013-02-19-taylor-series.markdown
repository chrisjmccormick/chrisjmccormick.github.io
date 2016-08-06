---
author: chrisjmccormick
comments: true
date: 2013-02-19 19:43:57+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5419
published: false
slug: taylor-series
title: Taylor Series
wordpress_id: 5419
---

I've seen the Taylor series referenced a number of times now in different vision concepts:



	
  * Optical Flow - The method for finding Optical Flow is based on taylor series approximations of the image signal. (http://en.wikipedia.org/wiki/Optical_flow)

	
  * SIFT - The Taylor expansion of an image is used to find subpixel coordinates for keypoints in SIFT (http://www.aishack.in/2010/05/sift-scale-invariant-feature-transform/4/).




### Resources


_Khan Academy_

Whenever a Khan Academy video is available on YouTube for learning a concept, it's a great place to start.

There appear to be a number of Khan Academy videos referencing the Taylor Series, I watched this one: http://www.youtube.com/watch?v=8SsC5st4LnI.

_Mathforum.org_

This wiki-style article looks interesting. It starts with a great animation to help illustrate the concept, and discusses some of the engineering significance of the Taylor Series.

http://mathforum.org/mathimages/index.php/Taylor_Series


### Description / My Notes


Here is the Taylor Series equation:

![f(a)+\frac {f'(a)}{1!} (x-a)+ \frac{f''(a)}{2!} (x-a)^2+\frac{f^{(3)}(a)}{3!}(x-a)^3+ \cdots. ](http://upload.wikimedia.org/math/d/8/f/d8f92ef8e46a567502e11ceb74574b40.png)

The first term ensures that for x = a, the result is equal to f(a). The second term ensures that the first derivative of the equation is equal to f '(a). The third term makes sure that the second derivative of the equation is equal to f ' '(a). And so on.



	
  * All of the derivatives of the Taylor Series are going to be equal to the derivatives of the original function, at least at the point 'a'.

	
  * If you carry out the equation to infinity, the result will actually be equal to the original function.


The factorial under each of the terms ensures that each additional term in the series has a successively smaller impact at the starting point ('a'). These terms come in to play more as you get farther from 'a'.
