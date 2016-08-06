---
author: chrisjmccormick
comments: true
date: 2013-04-12 23:32:17+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5579
published: false
slug: stanford-machine-learning-lecture-8
title: Stanford Machine Learning - Lecture 8
wordpress_id: 5579
---

The last lecture leaves off just before section 7 Kernels in the lecture notes.


## Kernels


Before Professor Ng goes into the discussion of Kernels, I think it's helpful to know that SVMs are able to handle non-linear separation boundaries by mapping the input vectors into a higher-dimensional space in which they are linearly separable.

An implication of this is that the vectors may end up being very high-dimensional after we perform this mapping. Currently, our equations involve computing a lot of inner products between the input vectors, and this can become very expensive.

The method of Kernels enables you to compute the product of two very high-dimensional vectors efficiently.

The idea is that for some feature mappings, there exists a kernel function which can efficiently compute the inner product of two feature mapped vectors. This is best illustrated by the examples in the lecture notes.

Professor Ng also shows how to verify whether a given function can be used as a kernel function. [? - 27:15]

Kernel functions are chosen for a problem based on how well they measure the difference between two examples in your problem. That is, the kernel function should give a large value for dissimilar inputs and a small value for similar inputs.

The Gaussian kernel that he shows actually corresponds to an infinite dimensional feature vector! The feature vector could never be explicitly represented, then, but the dot product between two of these vectors can be computed using the Gaussian kernel.


## Soft Margins


[35:40]


## Applications of SVMs


[1:09:20]



	
  * Polynomial kernel or Gaussian kernel works well for hand-written digit recognition.

	
    * Previously accomplished best by a Neural Network. Sounds like the Neural Network had knowledge of the arrangement of pixels (i.e., the ROI wasn't just represented as a vector...)




	
  * Classify a protein sequence example into different classes of proteins.

	
    * There are 20 amino acids

	
    * One of the challenges with this application is that the length of the sequence can vary largely.

	
      * For every possible arrangement of four amino acids, count how many times it occurs in the sequence.

	
      * There are 20^4 possible arrangements of four amino acids. 160K dimensional feature vector

	
      * There is an efficient "Dynamic Programming" algorithm which can compute the inner product between these very large vectors.








