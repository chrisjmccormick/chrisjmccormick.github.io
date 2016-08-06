---
author: chrisjmccormick
comments: true
date: 2013-03-13 17:58:33+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5465
published: false
slug: sift-object-recognition-tutorial
title: SIFT Object Recognition Tutorial
wordpress_id: 5465
---

The SIFT algorithm and its application to vision problems are complex topics incorporating many computer vision fundamentals. It is difficult to reach an intuitive understanding of these concepts without some background in computer vision.

Below are some of the concepts you will want to familiarize yourself with before attempting to understand SIFT.



	
  * Gradient vectors - what they are, what information they provide, and how they're computed.

	
  * The Gaussian filter - including how it is programmed and how it helps in edge detection.

	
  * The Laplacian of Gaussian filter - what it is and how it is used for edge detection.


**Overview**

The SIFT algorithm is not a single process for performing object recognition, but rather is a collection of solutions to independent problems such as interest point detection, feature description, feature matching, and finally object recognition. It's important to make this distinction because it's actually possible to substitute other implementations for each of these steps. Compartmentalizing the different steps like this helped me to wrap my head around each step--when you study SIFT key point detection, for example, it helps to know that the goal of that step is just to find good keypoints, and ignore the broader problem of object recognition.

The primary contribution of the SIFT algorithm is to the interest point detection and feature description steps. These are the two steps that you'll find the most explanation for online.

Let's look first at what's meant by each of these steps.

**Keypoint detection**

Keypoints are also referred to as "interest points". A keypoint is simply a location in an image where there is something unique; keypoints are typically corners.

The SIFT keypoint detection algorithm is the hardest part to understand, requiring the most background in math and vision.

A keypoint can be represented simply an x, y pixel coordinate. However,  in the process of locating good keypoints in an image, the keypoint detector will often compute some useful values which are saved along with the keypoint's location. This additional information can be useful in computing a "descriptor" for the keypoint, described next.

**Feature descriptor**

The pixels in the neighborhood of a keypoint represent an "image feature", a feature of an object which is unique and interesting and helpful in recognizing that object in other images. A binary representation of the feature is stored in what is referred to as a "feature descriptor". The simplest feature descriptor would just contain the pixel values of the neighboring pixels (perhaps in 16x16 square region around the keypoint, for example). SIFT defines a more clever descriptor, however, based on the gradient vectors of the neighboring pixels.

**Feature matching**

Feature matching is a simple process. You have the features which describe the object you're looking for, and you have a test image where you're trying to locate the object.  For each of your object's features, you look for the closest matching feature in the test image.

This is as far as most SIFT tutorials get, and when you look at the matching results, it can look like you've detected the object. You're not really there yet, though. If you knew for certain that the test image contained your object, then this step might be enough to help you locate it. This step isn't enough, though, to tell you whether the test image actually contains the object or not. You're going to find the "closest matches" for your object whether the object is present in the test image or not!

**Object recognition**

To actually say whether you've detected your object or not, you'll need an algorithm for determining whether the "closest matching features" that you've found actually correspond to your object. The Hough transform is applied here to determine if your set of feature matches are in the same relative positions to each other as they were in the test image.
