---
author: chrisjmccormick
comments: true
date: 2013-04-11 17:48:39+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5562
published: false
slug: stanford-machine-learning-lecture-7
title: Stanford Machine Learning - Lecture 7
wordpress_id: 5562
categories:
- Lecture Notes
- Stanford Machine Learning
---

This lecture covers:


## **Resources**


The YouTube [video](http://www.youtube.com/watch?v=s8B4A5ubw6c) for the lecture.

_Review Notes_

This lecture will talk a lot about optimization problems. There are some review notes on the site for convex optimization: [part 1](http://see.stanford.edu/materials/aimlcs229/cs229-cvxopt.pdf) and [part 2](http://see.stanford.edu/materials/aimlcs229/cs229-cvxopt2.pdf). (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

_Course Lecture Notes_

This lecture continues with the [third set of lecture notes](http://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf) on Support Vector Machines. (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

Again, I recommend going back and forth between the video and the lecture notes. After you watch a portion of the video, read the corresponding section in the lecture notes to ensure you understand it correctly.

_ Additional Lecture Notes_

Look at [Holehouse's page](http://www.holehouse.org/mlclass/) for notes on support vector machines.


## My Notes


**Maximum Margin Classifier**

****The lecture begins with a recap of the maximum margin classifier. He shows how the maximization problem can be re-written in a way that ensures it is convex, meaning that there is only one global optimum result. He doesn't discuss the optimization methodology, but explains that there is off-the-shelf software that you can get which will perform it for you...

**Dual Form - The Lagrangian**

FYI, I found this section very difficult to get through. We're going to apply the method of Lagrange multipliers to re-write our optimization problem. By the end of the lecture, we'll have a new form for the optimization problem in which we need to solve for the Lagrange multipliers (denoted alpha-sub-i) instead of _w _and _b._

This new form of the optimization problem will have some important implications which make it efficient to solve.

Lagrange multipliers and the Lagrangian are just a method for finding the min or max of a function _f _with  additional constraints _h _and _g_. Instead of directly solving your original optimization problem, you construct an equation called the Lagrangian which is basically the sum of your original function and the constraint equation:

[![SVM_Lagrangian](http://chrisjmccormick.files.wordpress.com/2013/04/svm_lagrangian.png?w=470)](http://chrisjmccormick.files.wordpress.com/2013/04/svm_lagrangian.png)

In this equation:



	
  * _g_(_w_) is a function which must be less than or equal to 0.


	
  * _h_(_w_) is a function which must be equal to 0.

	
  * Each data point in our training set is going to supply a constraint equation; the subscript 'i' is the constraint equation given by the _i_th training example.

	
  * Each constraint equation has a coefficient associated with it which we must find (for example, each data point will its own value of alpha).


For this problem, we won't have an _h_(x) constraint equation, just a _g_(x) constraint equation. The _g_(x) constraint is taken from our previous optimization work:

[![SVM_g(x)_constraint](http://chrisjmccormick.files.wordpress.com/2013/04/svm_gx_constraint.png)](http://chrisjmccormick.files.wordpress.com/2013/04/svm_gx_constraint.png)

We originally made the constraint that the functional margin of the data set must be 1. This is the same as saying that the functional margin of the data point closest to the separation boundary (which has the smallest functional margin and defines the functional margin of the set) must be 1. The _g_(_w_) term above is just a reorganization of this constraint.

Think about what has to happen with each of these terms if we want to maximize the Lagrangian.

For the _g _term, we have the constraints that _g_(_w_) must be 0 or negative, and alpha must be 0 or positive. If alpha is positive and g is negative, that's not going to help us maximize this function. Either alpha or g or both need to be 0 in this term to maximize the function.

For the _h_ term, we already have the constraint that _h_ is equal to 0; that whole term will equal 0.

So if the constraints are met, the second and thirds terms will just be zero and we really just need to maximize _f._

Later in the notes, he points out that in order for g(_w_)_ _to be 0, the functional margin must be equal to 1. It follows from this that for any non-zero alpha value, the functional margin must be equal to 1. This implies that only the data point(s) closest to the margin will have a non-zero alpha value. This is significant, because it means that all of the terms from the other data points will disappear, and we really only need to keep the data points whose functional margin equals 1.

An important point about d* and p*: under certain conditions, they are equal, and d* may be much easier to solve.

Check out this guy's blog: http://richardminerich.com/2012/12/my-education-in-machine-learning-via-cousera/

Also look at this guy's blog: http://realizationsinbiostatistics.blogspot.com/2011/12/statisticians-view-on-stanfords-public.html

Holehouse notes: http://www.holehouse.org/mlclass/
