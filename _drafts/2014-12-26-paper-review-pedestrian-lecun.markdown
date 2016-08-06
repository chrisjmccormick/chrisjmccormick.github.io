---
author: chrisjmccormick
comments: true
date: 2014-12-26 22:23:01+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=6009
published: false
slug: paper-review-pedestrian-lecun
title: Paper Review - Pedestrian Detection with Unsupervised Multi-Stage Feature Learning
wordpress_id: 6009
---

In this post I'm simply sharing my notes on the paper _Pedestrian Detection with __Unsupervised Multi-Stage Feature Learning_ published in 2013 by Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and Yann LeCun, all at NYU.

Deep neural networks, especially in the form of Convolutional Neural Networks, have been making large strides in computer vision and speech. They are able to perform richer feature extraction than even the most clever hand-coded feature extraction algorithms. So it makes sense that someone should be able to build a CNN that outperforms the state-of-the-art pedestrian detection feature extraction techniques.

Yann LeCun is the father of Convolutional Neural Networks and a major figure in deep learning, so the fact that this paper is coming from his lab adds a lot of credibility to it, I think.  As of this posting, the paper has already been cited by 60 other papers according to Google Scholar.

**1. Introduction**

"While low-level features can be designed by hand with good success, mid-level features that combine low-level features are difficult to engineer without the help of some sort of learning procedure."



	
  * They're arguing that it's really the higher layers of the CNN (which extract more complex features) which provide the competitive advantage.


"The system uses unsupervised convolutional sparse auto-encoders to pre-train features at all levels from the relatively small INRIA dataset [5], and end-to-end supervised training to train the classifier and fine-tune the features in an integrated fashion."

	
  * Stanford's [deep learning tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) provides an excellent introduction to sparse auto-encoders.

	
  * Stanford's tutorial also explains how merely using auto-encoders to train the individual layers alone does not yield great performance--the key is to then fine-tune the network with supervised back propagation training.


"Additionally, multi-stage features with layer-skipping connections enable output stages to combine global shape detectors with local motif detectors."

	
  * I'll have to read further to understand this fully. However, I know that it can be useful to include lower level features in the final feature vector.


**2. Learning Feature Hierarchies**

**2.1. Hierarchical Model**

"Each layer of the unsupervised model contains a convolutional sparse coding algorithm and a predictor function that can be used for fast inference."



	
  * I am less familiar with sparse coding and the terminology used there, though of course the intent is the same--to discover a set of useful features. I am not sure what is meant here by a "predictor function" or "fast inference"...

	
  * 

