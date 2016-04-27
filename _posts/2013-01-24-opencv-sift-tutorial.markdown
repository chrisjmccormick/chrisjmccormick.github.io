---
author: chrisjmccormick
comments: true
date: 2013-01-24 21:10:46 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/01/24/opencv-sift-tutorial/
slug: opencv-sift-tutorial
title: OpenCV SIFT Tutorial
wordpress_id: 5380
tags:
- matcher_simple
- OpenCV
- SIFT
---

This tutorial covers SIFT feature extraction, and matching SIFT features between two images using OpenCV's 'matcher_simple' example. It does not go as far, though, as setting up an object recognition demo, where you can identify a trained object in any image.


## OpenCV Setup & Project


If you haven't already, get OpenCV installed and a project setup in Visual Studio. You can find some instructions for doing that [here](http://chrisjmccormick.wordpress.com/2013/01/24/opencv-setup-in-visual-studio-2010/).

I've also uploaded my Visual Studio project, [here](https://docs.google.com/file/d/0B-kWgXJRQkQ7Z3NLa1RzYzlsZUU/edit). It assumes you have OpenCV 2.4.3 installed to C:\opencv\, and that you've added OpenCV to your path as described in the instructions I linked to above. If you have a different version you'll have to change the referenced library names.


## 'matcher_simple.cpp' Example


If you look under "C:\opencv\samples\cpp\" you'll find a big disorganized mess of example source files and images.

One of the examples, 'matcher_simple.cpp', provides an introduction to feature extraction. It uses SURF by default, but you can change it to SIFT with a simple find-and-replace of 'Surf' with 'Sift'!

Here is the only documentation I've been able to find on the example:

http://docs.opencv.org/doc/user_guide/ug_features2d.html

The documentation doesn't provide much insight into the significance of the example, and disappointingly doesn't offer any recommendations for example images to play with.

Here's what the matcher_simple example shows you how to do, though:



	
  1. Load two images

	
  2. Detect keypoints in each image

	
  3. Extract SIFT features for those keypoints

	
  4. Find the closest matching features between the two images

	
  5. Display the images side-by-side and draw lines connecting the matching features


I have a sequence of images that I've captured of me holding a flash card in different positions and angles. The flash card has a penguin on it. Here are the two frames from the sequence that I've used for this example.

[![frame_18](http://chrisjmccormick.files.wordpress.com/2013/01/frame_18.png)](http://chrisjmccormick.files.wordpress.com/2013/01/frame_18.png) [![frame_20](http://chrisjmccormick.files.wordpress.com/2013/01/frame_20.png)](http://chrisjmccormick.files.wordpress.com/2013/01/frame_20.png)

I took two images from the sequence, and cropped one of them down to just the penguin, then ran the example on the two images. Below is the result.

[![SiftMatching](http://chrisjmccormick.files.wordpress.com/2013/01/siftmatching.png)](http://chrisjmccormick.files.wordpress.com/2013/01/siftmatching.png)

The example code is extracting all of the SIFT features that it can find in both images. Then, for each feature in the left image, it's finding the closest matching feature in the image on the right. It draws a line from each keypoint in the left image to its closest match in the right image. I can see about five features that are clearly matched incorrectly, but it looks like the majority of them are correct.

You can probably see why I cropped one of the images. If I had two full images, it would be matching features that I didn't care about, like those on the shutters in the background.


## Object Detection With SIFT


Simply matching the features between two images is a good illustration of how SIFT works, but it doesn't get you all the way to actually recognizing a trained object. For example, I'd like to set up an example that is able to reliably detect the penguin flash card in a video stream. When I've found some example code, I'll write another post.
