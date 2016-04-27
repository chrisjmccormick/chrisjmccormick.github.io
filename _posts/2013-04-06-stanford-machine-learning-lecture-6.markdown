---
author: chrisjmccormick
comments: true
date: 2013-04-06 20:00:35 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/04/06/stanford-machine-learning-lecture-6/
slug: stanford-machine-learning-lecture-6
title: Stanford Machine Learning - Lecture 6
wordpress_id: 5550
tags:
- Functional Margin
- Geometric Margin
- Machine Learning
- Neural Networks
- Stanford
- Support Vector Machines
---

This lecture covers:



	
  * A variation to the Naive Bayes text classifier which performs better for spam filtering.

	
  * A _brief_ discussion of Neural Networks

	
  * Beginning of the discussion on Support Vector Machines

	
    * Functional and Geometric Margins







## **Resources**


The YouTube [video](http://www.youtube.com/watch?v=qyyJKd-zXRE) for the lecture.

_Review Notes_

__For once, you shouldn't need any additional review material for this lecture.

_Course Lecture Notes_

This lecture wraps up the [second set of lecture notes](http://see.stanford.edu/materials/aimlcs229/cs229-notes2.pdf), and starts the [third set of lecture notes](http://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf). (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

Again, I recommend going back and forth between the video and the lecture notes. After you watch a portion of the video, read the corresponding section in the lecture notes to ensure you understand it correctly.

_ Additional Lecture Notes_

It finally occurred to me in this lecture that the Machine Learning course offered through Corsera must be a little different than the one available through SEE and YouTube.

Most notably, Alex Holehouse's notes go into a lot of depth on neural networks, while Professor Ng only briefly mentions neural networks in the YouTube video.

At any rate, take a look at [Holehouse's page](http://www.holehouse.org/mlclass/) for notes on neural networks and support vector machines.


## My Notes


**Naive Bayes**

It is possible to use Naive Bayes for variables which have more than just two values. You can even apply it to continuous variables by discretizing them (map ranges of the value into buckets).

**Naive Bayes with Multinomial Event Model**

Some words are likely to occur multiple times in an e-mail, and we'd like to incorporate that into the decision.

Before, we were looking at the probability that a spam e-mail contains a given word based on the number of spam e-mails in our training set which contain that word.

Now, we're looking at the probability that the word at a given position in a spam e-mail is a particular word, based on the number of times that word occurred in all of our spam training e-mails. For example, count the number of times "Viagra" appears in all of your training spam e-mails, and divide it by the total number of words in all the spam e-mails, and that's p(x | y) for the word "Viagra".

For text classification, this second model always seems to perform better than the first model presented in the previous lecture.

**Nonlinear Classifiers - Neural Networks**

[25:00]

Sometimes the classes can't be separated by a straight line.

Back propagation is just the neural network term for gradient descent.

Professor Ng doesn't use NN, there seems to be some consensus that SVMs are a better off the shelf algorithm. The challenge with NNs is the optimization--there tend to be lots of local minima that you can get stuck in, and so it's a hard optimization problem.

**Support Vector Machines**

The discussion will start with linear classification, and get towards nonlinear classification.

Two important intuitions about classification:



	
  * "It would be really nice" if we were very confident in our classifications - Functional Margin

	
  * "It would be really nice" if all of our training examples were very far from the decision boundary - Geometric Margin


**Functional Margin**

To understand the functional margin, I think it helps to look a little at the meaning of the linear function, theta transpose x, in logistic regression. Lets take the example of classifying a tumor based on two variables, patient's age and tumor size. If you equate theta-transpose x to zero and plot it, you get a line splitting the two classes--the decision boundary. If you plot theta transpose x directly, though, you get a plane, which has positive values for malignant tumors (y = 1) and negative values for benign tumors (y = 0). I find it helpful to think of this as a scoring function, where the score becomes increasing positive as you move from the decision boundary in the direction of the y = 1 class, and increasing negative as you move in the other direction from the boundary.

The functional margin is just saying that ideally you would like the "score" for all of your y = 1 data points to be a very large positive value, and the "score" for all of your y = 0 data points to be a very large negative value.

When computing the functional margin for a data set:



	
  * We normalize the vector of weights to have a magnitude of 1

	
  * We take the functional margin of the data point closest to the decision boundary as the function marginal of the whole data set. In other words, we take the "worst case" functional margin as the functional margin of the data set.


**Geometric Margin**

The geometric distance of a training example is just its distance from the decision boundary. See [http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line](http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)

The notes claim that if you plot the vector of weights w, it_ _will be perpendicular to the decision boundary. For example, take z = 2x + 2y. Set z = 0 and solve for y and you get the decision boundary, y = -x. The weight vector is [2  2], has an angle of 45 degrees, which is perpendicular to y = -x.

In the algebra that he goes through, it helps to clarify that gamma (i) is now the geometric margin (not the functional!), and is a scalar.

Again, we define the geometric margin of a dataset to be the distance between the decision boundary and the closest training point.

As we search for our parameters _w_, we are going to enforce that ||_w|| = _1. As the lecture notes show, this has the nice property of making the functional margin equal to the geometric margin. This is good, because now we only need to optimize _w _and _b _to maximize  one of those two properties. So we'll just maximize the geometric margin, and you can forget about how the parameter choices are affecting the functional margin, because they're equivalent!

**Maximum Margin Classifier**

Our choices of _w _and _b_ will affect the slope and the placement of the decision boundary line. We want to position it and angle it to give the largest possible margin for both classes. Pretty intuitive! If you start thinking about how the parameters affect the slope of the hyperplane, it gets a little more confusing, but by enforcing ||_w_|| = 1 we don't have to worry about what the hyperplane is doing.

The first condition of the maximization problem made me want to pull my hair out at first. Maximize gamma, parameterized by gamma, such that the equation for gamma is greater than or equal to gamma!?!

**![](https://lh4.googleusercontent.com/CiS-p3h0iVZVEJdifXsrYpss9H3Ng5bFKzt5MZ0y7gPfnYvQWObbUIOdlbyiU13brLGE8YZD7KL2oPKKu7L68I-_skq-7XFNo5hYqycS2oBGtT8YcR4JjZm7)**

What this is saying is actually pretty simple--we specifically want to maximize the worst-case geometric margin for our data set.

Every data point in the training set is going to have its own geometric margin value. In the above equation, gamma refers to the functional margin of the whole data set, whereas the equation to the left of the >= is the equation for the geometric margin of data point _i_.

At this point, what we have is something about as effective as the logistic regression classifier.


