---
author: chrisjmccormick
comments: true
date: 2013-01-24 18:29:03+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=33
published: false
slug: hough-transform-tutorial
title: Hough Transform Tutorial
wordpress_id: 33
tags:
- Hough Transform
---

The Hough Transform (pronounced “Huff”) is used to recognize lines and shapes in an image. Even when you’ve identified specific pixels in an image as belonging to an edge, it’s still a non-trivial problem to actually recognize the shape and position of the line that they fall on. That’s where the Hough Transform comes in--it helps fit the data points to an equation.

Discussions of the Hough Transform typically start with the simple case of straight lines, and then expand from there, so that’s what we’ll do as well.

**Why The Hough Transform?**
One common method for fitting data points to a line is the least squares method. It takes the intuitive approach of finding (by solving equations) the line which minimizes the “error” or distance between the line and all of the data points.

But what happens when the pixels actually form two intersecting lines? The least squares method will give a model that's basically an average of the two, and pretty useless.

This is an important strength of the Hough Transform--it can actually help recognize that there are multiple lines to be fitted. How is that possible?

The Hough Transform uses a voting approach to determine the best model. So you tally up all of the votes to find the candidate with the most votes. But then, you can also look at the runner up. If the second best candidate is a very different model, but has almost as many votes, then it's likely that both models are of interest.

In general, when an algorithm involves a voting step, there's the opportunity for multiple candidates to be selected. (Another example is keypoint selection in the SIFT algorithm, where there's a similar voting step that allows for the possibility of multiple keypoints being generated from one pixel).

**
****Resources
**Wikipedia: **
**[http://en.wikipedia.org/wiki/Hough_transform](http://en.wikipedia.org/wiki/Hough_transform)
The tendency for computer vision articles on Wikipedia is to be factual but not necessarily communicate the concepts well. I recommend it as a reference but maybe not as an introduction.

UCF Lecture:
[http://www.youtube.com/watch?v=hYcugbbf9ug](http://www.youtube.com/watch?v=hYcugbbf9ug)
This lecture provides a decent introduction. There are a few points where he completely lost me and I had to try and find other sources to explain his points.
He spends about 10 minutes going over some common line fitting techniques before moving on to the Hough Transform.

He also covers the "generalized Hough transform" (applying the Hough Transform to arbitrary shapes).

UofE Tutorial:

http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm

This page offers a tutorial of the Hough Transform, and may be an even better starting point. I'm still reading through it.

**High Level Description**

Let's start with the simple example of fitting data points to a straight line.

I think it helps to think of the Hough transform as a brute force approach to finding the best fitting line. All the Hough transform does is look at all of the possible lines which could intersect each individual data point, and then chooses whichever line occurs most often across all of the data points.

Well, wait a minute, there are an infinite number of lines that could intersect a given point, right? Correct, so the engineer has to choose some increment value to change the lines by. For example, you could rotate the line by 1 degree at a time, so that there are only 360 possible lines that could intersect a given point.

That first data point casts one vote each for each of the 360 possible lines that intersect it. Repeat this for every edge pixel you have. Once all of the edge pixels have cast their votes, just take the line with the most votes!

To look at how you'd actually implement this, we'll have to go through the math more formally.

**Fitting A Straight Line**

Let's look in detail at how the Hough Transform is used to fit a set of coordinates to a line.

The below image is taken from Dr. Shah's slides in the YouTube lecture referenced above.

[![Image](http://chrisjmccormick.files.wordpress.com/2013/01/houghtransform_mc_space.png?w=406)](http://chrisjmccormick.files.wordpress.com/2013/01/houghtransform_mc_space.png)

On the left we have five data points, plotted in the (x, y) space.

Let's consider just one data point on its own, say data point 1. There are an infinite number of lines which could intersect that data point. However, for a given slope value, there is only one y-intercept value that will create a line which intercepts that point. Put another way, f**or the equation for a line, y = mx + c, if you take only one (x, y) data point, that’s not enough to solve for m and c. However, for a given m, there is only one possible value of c.**

The right hand of the diagram shows all of the possible pairings of m and c for each data point. Line 1, for example, represents all of the possible combinations of m and c that intersect data point 1.  **Just as one set of parameters (m, c) gives you a straight line in (x, y) space, one (x, y) data point gives you a straight line in (m, c) space.**

**To find the best fitting line for the data points, you would just need to find the point (m, c) where the most lines intersect. This is where the voting comes in. If two lines intersect at a particular (m, c) point, that's 2 votes, if three lines all intersect at the same (m, c) point, that's three votes, etc.**

Note: If all of the points are close to the best fit line, but none actually fall exactly on the best fit line, then you'll never get more than two lines intersecting at a particular (m, c) point. There may be a bunch of intersections all in the neighborhood of a single (m, c) point, but no one specific point is going to accumulate more than 2 votes.

This is the problem of noise. You handle this by choosing an increment size. The increment size will cause lines which are similar (though not identical) to be grouped together. Too small of an increment size makes it susceptible to noise (no line will acumula, while too large of one can lead it to group multiple lines together when they are really separate.
