---
author: chrisjmccormick
comments: true
date: 2013-04-04 18:38:33+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5505
published: false
slug: stanford-machine-learning-lecture-4
title: Stanford Machine Learning - Lecture 4
wordpress_id: 5505
categories:
- Lecture Notes
- Stanford Machine Learning
tags:
- Lecture Notes
- Machine Learning
- Stanford
---


	
  * Logistic regression (lecture 3)

	
    * Newton's method




	
  * Exponential family

	
  * Generalized linear models


**Newton's Method**

Another method for finding the parameters to fit a line. It will often run faster than gradient descent because it exhibits "quadratic convergence"--it converges faster than gradient descent.

Professor Ng shows how Newton's method can find the point in a function where the value is 0. We also know that to maximize a function, you take it's derivative and set it 0. So Newton's method can also be used to find the maximum of a function _l_ by applying to it to the derivative of _l_.

The trade-off to Newton's method is that computing it requires computing the inverse of the Hessian matrix (second derivatives), which is an n x n matrix where n is the number of features. Matrix inversions are expensive, so for a large number of features this can be a problem.

**Generalized Linear Models **[16:40]

I skipped this part of the lecture (left off around 25:12).
