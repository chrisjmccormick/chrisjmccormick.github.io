---
author: chrisjmccormick
comments: true
date: 2014-05-25 05:16:38 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2014/05/25/stanford-deep-learning-tutorial/
slug: stanford-deep-learning-tutorial
title: Stanford Deep Learning Tutorial
wordpress_id: 5827
tags:
- Deep Learning
- Feature Extraction
- Machine Learning
- Unsupervised Feature Learning
---

Stanford has a very nice [tutorial on Deep Learning](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) that I've read through, and I'm in the process of going through it in more detail and completing the exercises. I'll be posting my notes on each section as I go.


### Why Deep Learning?


My understanding of the significance of Deep Learning is still evolving, but here are some of the high level points, as I currently understand it.

_Deep Networks_

One way to look at deep learning is as an approach for effectively training a Multilayer Perceptron (MLP) neural network with multiple hidden layers. A "deep" MLP.  (Note: The MLP architecture is what's most commonly meant by "neural network", and is the architecture taught in the Coursera course on Machine Learning).

Given enough neurons, a 3-layer MLP (that is, an MLP with a single layer of hidden neurons) is capable of doing anything. It can approximate any function, or define any arbitrary decision boundary for classification. Given that, why would we ever want to train an MLP with more than one hidden layer? The answer is that the same amount of complexity can be accomplished with fewer neurons if you use multiple hidden layers.

According to the tutorial, there are some difficult issues with training a "deep" MLP, though, using the standard back propagation approach. So one way to view deep learning is as a solution to the problem of training deep networks, and thereby unlocking their awesome potential.

_Unsupervised Feature Learning_

The other exciting aspect of these techniques is the ability to learn powerful feature extraction techniques using only unlabeled training data.

If you follow my blog, you may know that I've spent a fair amount of time researching person detection using the Histogram of Oriented Gradients (HOG) approach. This is a perfect example of the challenge in machine learning that deep learning may address.

First, the HOG feature extraction algorithm had to be carefully and cleverly designed by researchers. Navneet Dalal is famous in the computer vision community for his work on HOG; the HOG algorithm is a big deal. The techniques in this deep learning tutorial point at a methodology for _learning_ feature extraction algorithms from unlabeled data, without requiring clever engineers like Dalal to hand design the algorithm.

Second, the machine learning classififier (a linear SVM in the case of the original HOG detector) must be trained on a large amount of hand-labeled training data in order to recognize people. The deep learning approach can learn from unlabeled data, which is obviously much more abundant.


### The Tutorial


Stanford's deep learning tutorial seems to be structured like a course, with programming assignments in Octave / Matlab for each section.

Andrew Ng contributed to this tutorial, and it largely uses the same notation and conventions as his Coursera course, so that's pretty nice if you (like myself) learned Neural Networks through his course.

The tutorial is laid out as a list of Wiki pages organized into sections.  At the end of each section is a programming exercise. The wiki format is a little confusing at first for a tutorial, since it's just a series of links to articles (with no forward and back buttons in the articles for advancing to the next topic), but the website ultimately does appear to be intended as a tutorial to be read sequentially.


