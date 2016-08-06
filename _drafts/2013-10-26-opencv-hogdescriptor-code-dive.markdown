---
author: chrisjmccormick
comments: true
date: 2013-10-26 17:03:03+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5708
published: false
slug: opencv-hogdescriptor-code-dive
title: OpenCV HOGDescriptor Code Dive
wordpress_id: 5708
---

Recently, I've been picking apart the OpenCV HOG detector looking for additional tricks and ideas that I could apply to my own implementation.

The HOGDescriptor structure is defined in:

C:\opencv_2_4_5\modules\objdetect\include\opencv2\objdetect\objdetect.hpp

The source for the detector is in:

C:\opencv_2_4_5\modules\objdetect\src\hog.cpp

The main function of interest is 'HOGDescriptor::detectMultiScale', which you use to search an image for people.


## Result Clustering / Non-Maximum Suppression


The last step of the HOG detector image search is to try and group search results which correspond to the same detection at multiple image scales. The idea is to end up with only a single detection per-person, and hopefully to eliminate some false positives which are only recognized at one or two scales.

This is done using the 'groupRectangles' function definied in objdetect\cascadedetect.cpp. The 'groupRectangles' uses the OpenCV 'partition' function for cluster assignment, passing it an instance of the 'SimilarRects' class for comparing two rectangles and determining if they represent a single detection.
