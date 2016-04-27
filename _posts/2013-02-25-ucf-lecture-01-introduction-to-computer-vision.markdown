---
author: chrisjmccormick
comments: true
date: 2013-02-25 19:09:29 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/02/25/ucf-lecture-01-introduction-to-computer-vision/
slug: ucf-lecture-01-introduction-to-computer-vision
title: UCF Lecture 01 - Introduction To Computer Vision
wordpress_id: 5436
---

[Lecture 01 Introduction to Computer Vision](http://www.youtube.com/watch?v=715uLCHt4jE)

This lecture covers the very basics of a what a digital image or digital video is. It also provides a brief overview of the main topics and applications in computer vision, without going into much detail about algorithms or implementations. I found it valuable to hear about what are currently the hot topics and research areas in computer vision.

My notes from the lecture:



	
  * He discusses what constitutes an image (2D array of pixels with values 0 - 255)

	
  * He discusses how an image is formed: projection of the a 3D object onto a 2D image plane.

	
  * Discusses approaches for reconstructing 3D information from 2D images.

	
    * Stereo - Depth information from two cameras.

	
    * Shading - makeup fools the human brain into giving your face a different shape.

	
    * Texture - Texture is a repeated pattern. You can look at distortions in the pattern to recover 3D information.

	
    * Shape from Motion - Looking at just a small collection of moving dots, we can make out that it’s a person based on the motion.




	
  * He recommends a book on computer vision by Rick Szeliski’s, a principal researcher at Microsoft research.

	
  * He shows a demo video of Microsoft’s Photosynth, which attempts 3D constructions of scenes from 2D images gathered on the web.

	
  * Shows some example applications of computer vision:

	
    * Mosaic - Stitching together images from a video sequence to construct a complete view of the scene.

	
      * One example is video from UAV tracking a car down a road, mosaic stitches together all of the images of the road to create more of a map of the area.




	
    * “Human Detection” - Does the frame contain a person?

	
    * Airplane detection

	
    * Face Recognition

	
    * Facial Expressions

	
    * Detecting Driver Alertness

	
    * Lip Reading - Our brain supplements audio with lip reading.

	
    * Video Surveillance and Monitoring

	
      * Automated Surveillance System - Detection & Tracking




	
    * They are working on a project for airport surveillance. Multiple high resolution cameras providing 360 degree view. Called wide-area surveillance (WAS), lots of people in the airport.

	
      * Homeland Security Advanced Research Project Agency - HSARPA

	
      * Called NONA system, couldn’t find links online though




	
    * UAV Surveillance

	
      * Currently the surveillance footage is reviewed by humans, because we don’t have the techniques to analyze these with a lot accuracy.

	
      * Part of the challenge is that you need to remove camera motion from the equation.




	
    * Unmanned Ground Vehicle (UGV) - Self driving cars.

	
    * Human Action Recognition - Recognizing the actions, activities that people are doing.

	
      * Weizmann Action Dataset - A collection of videos constituting 9 actors and 10 actions. Try to figure out which action the person is performing.




	
    * Accurate Image Localization - “Where Am I?”

	
    * Layer Based Video Composition - Remove a foreground object from a video, filling it in with background information acquired over the sequence of frames. This is used by the film industry? Also background replacement.

	
    * 




