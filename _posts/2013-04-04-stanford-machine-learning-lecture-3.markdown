---
author: chrisjmccormick
comments: true
date: 2013-04-04 17:05:58 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/04/04/stanford-machine-learning-lecture-3/
slug: stanford-machine-learning-lecture-3
title: Stanford Machine Learning - Lecture 3
wordpress_id: 5495
tags:
- Lecture Notes
- Locally Weighted Regression
- LOESS
- Logistic Regression
- LOWESS
- Machine Learning
- Stanford
---

The third lecture covers the following topics (except where noted):



	
  * Linear regression (lecture 2)

	
    * Locally weigthed regression




	
  * Probabilistic interpretation

	
  * Logistic regression

	
    * Digression perceptron




	
  * Newton's method (lecture 4)




## **Resources**


The YouTube [video](http://www.youtube.com/watch?v=HZ4cvaztQEs) for class.

_Probability Theory Review_

This lecture will use some probability theory. Some review notes on probability theory are provided [here](http://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf) (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

The probability theory is used to demonstrate the correctness of using least squares as a cost function. If you just want to take it for granted, though, I'm not sure this part of the lecture is very important.

_Course Lecture Notes_

This lecture still uses the first set of [lecture notes](http://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf). (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

Again, I recommend going back and forth between the video and the lecture notes. After you watch a portion of the video, read the correspodning section in the lecture notes to ensure you understand it correctly.

_ Additional Lecture Notes_

Alex Holehouse's [notes on this lecture](http://www.holehouse.org/mlclass/06_Logistic_Regression.html) only cover logistic regression. He adds a lot to the discussion of that topic, though, so I highly recommend reviewing his notes.


## My Notes


**Polynomial Hypothesis Functions**

Your hypothesis function could also be a quadratic curve, and the gradient descent algorithm still works.

The gradient descent function is based on the derivative of the cost function (least mean square):

**![](https://lh3.googleusercontent.com/a-c2kLC1AVpCsjGb57_gp7NovKd9W-TOgCBK5jjBAKDYGVMQofyejNG_riwxb6bRm3ca_zirX64sRx44yfajsvVlSeoOHJvRITyNaQ_QYiefAq0Cigvnn4jY)**

So your hypothesis function could be:

**![](https://lh5.googleusercontent.com/JNy5ASDKKlzE9c3FCLGOKezkCNRTksIqSoB72gX7Dti3bzq775bJap64jNqRlFDfDBK7TI4BJ1gmIzBwpdJYxjbPWFSGWSJ1y-KD-fMN_rpCqi-1g5j8x5jV)**

And the derivative with respect to theta is unchanged.

We still end up with:

**![](https://lh3.googleusercontent.com/xQKURIvP42K1cBtADxKf4gUs5gG5K2WJUNlf4WaXrO40s63GtC4JRx_DKEBKQ00AK9UO0hqvWygMdhr7hi3nJ_klkHOVL7nuHl1r3tvx20v7JbAXCuX78BBs)**

This is significant because the data in our application may have a different model to it. For example, the model for home prices based on square footage may be more a quadratic (the housing prices start to plateau).

Professor Ng points out that with a high enough order polynomial, you can fit the data points exactly, but that may not really be ideal.--this is the concept of underfitting and overfitting. Fitting a straight line to data with an obvious quadratic component is called "underfitting", and fitting the data to a high order polynomial is called "overfitting".

This, in general, is the problem of feature selection.

**Parametric Learning Algorithm**

Defined as having a fixed number of parameters that are fit to the data (the theta values). Gradient descent is an example.

**Non-Parametric Learning Algorithm**

****Defined as:



	
  * # of parameters grows (usually linearly) with the size of the training set.

	
    * The algorithm is going to keep around the training set; the training set will be used directly.

	
    * KNN (k-Nearest Neighbors) is a simple example.





**Locally Weighted Regression**



	
  * A non-parametric learning algorithm.

	
  * Also referred to as "Loess" or "lowess" (a combination of syllables from the real name?)

	
  * Professor Ng mentions early in the lecture that this algorithm is one of his mentor's favorites.


How do you fit a function to the following data?

**![](https://lh4.googleusercontent.com/4LJSmTJrtiooDYKI2lFrzKz8mku6-j91YB1fS_QifB8UcqmAayBeb7aHvdjRoWMxff46H1kqghy_1IVZ7eE_fEUBjBs8OxL6rGCQ8SwviPzwHl6nl-ZMuXKy)**

One approach would be to fiddle with different features and eventually come up with something that fits well.

But Loess works as follows.



	
  * Take a single point, and only look at the datapoints in the vicinity.

	
  * Apply linear regression to fit a line to just this subset.


When we do the line fitting, however, we're going to modify our cost function. We're going to apply a unique weight value to each of the training examples. The weight value is given by an exponential equation, which causes training examples farther from our point of interest to contribute exponentially less to the error. The exponential equation is the gaussian function, though it's use here isn't related to probabilities. The gamma of the gaussian function can be used to control the width of the bell curve.

Professor Ng mentions that they've used this algorithm to successfully model the dynamics of a helicopter.

So it's not really correct to use the term 'subset' above--we're still using every single training example, we just apply a very small weight to the ones far from our point of interest.

A student makes the point that it doesn't seem like you're actually creating a model, since you are still keeping the entire data set. Even if you have the whole data set, though, you still need a way to come up with a hypothesis value for an input value that doesn't exist in your training set. So you are making a hypothesis based on the neighbors of the data point.

The Wikipedia page for [Local Regression](http://en.wikipedia.org/wiki/Local_regression) has some interesting insights about the algorithm. In particular, it sounds like the primary advantage of LOESS is that you don't have to choose a model, and the main disadvantage is the high compute cost.

Check out the graph midway through the following article (referenced from the above Wikipedia page), which shows an opinion poll using LOESS vs. linear regression:

[http://fivethirtyeight.blogs.nytimes.com/2013/03/26/how-opinion-on-same-sex-marriage-is-changing-and-what-it-means/?hp](http://fivethirtyeight.blogs.nytimes.com/2013/03/26/how-opinion-on-same-sex-marriage-is-changing-and-what-it-means/?hp)

Also, this article provides an example of applying local regression to voting polls:

[http://voteforamerica.net/Docs/Local%20Regression.pdf](http://voteforamerica.net/Docs/Local%20Regression.pdf)

The most interesting insight from that second link, to me, was the fact that the weights are applied using a matrix with the same size as the training data. This means, as I mentioned above, that every value in the data set actually contributes to the local regression (though we saw that the weights minimize the effect of examples farther away from our interest point).

**Probabilistic Interpretation**

He moves on to the probabilistic interpretation around [28:20]

Why minimize the sum of squares error, instead of some other metric?

Assume that the error is represented by a gaussian function. You want to choose the parameters theta which minimize that error.

He shows that the function representing likelihood of getting the output values y as a function of theta can be refactored to look just like the least squares function.

One important conclusion from this derivation is that the value of sigma (the variance) doesn't affect the cost function. Meaning it doesn't matter how wide the error distribution is. Apparently this fact will be used again later.

**Classification**

Examples:



	
  * Medical diagnosis

	
  * Will this house be sold in the next six months?

	
  * Is this e-mail spam or not?

	
  * Will this computing cluster crash within the next 24 hours?


**Logistic Regression**

For linear regression (fitting a line to our data), we used the hypothesis function:

**![](https://lh6.googleusercontent.com/AehCDUFstMFx4MCIitx_YrcLbLcqd0kjKi5Xa-H4jhVIi4J-D5ZsDEZAjs0rMbDyVt492pIi09aTzewQsQu4EADgc98Q3KH5O5WLGVCLbcqvIqE0_EkoOiBx)**

For logistic regression, we are going to plug that into a sigmoid to create a new hypothesis function. It still involves the linear combination of the inputs, but the sigmoid will cause the output to vary between 0 and 1.

**![](https://lh5.googleusercontent.com/sP0s_MQQv2rgMpsK0VNejOslrH-klYGrgDi25NSa36sVAzjgZKnKxwhikAPCb6-kOlMF1NArx5V3Mbz-imyZggYuZ0RRBDZG3XPMcLWns5mtJ4mjUrJ7Q0a3)**

It turns out we can find the parameters theta in exactly the same way we did for linear regression, using the gradient descent algorithm:

**![](https://lh3.googleusercontent.com/MadNre9FmUDmL-AG0pjtv15Imjtl-iNW1sDZmloffc7iPxcIWiMeYMlpDtlfseVLRg4MXw3txzHBw7SfsQEsXAd02VxDhKfhrbP0atw0iqrdO-eqtdbeI7gO)**

The difference is just in the form of our hypothesis function.

Note that even though you can see how this can be used to perform classification, we're still performing regression. Our training example values are discreet; for example, 0 for benign cancer and 1 for malignant cancer.  However, we are still fitting a continuous valued function to the data. So we're still essentially trying to find the function of a line which fits the data! The problem hasn't changed.

Try typing the following into Google: "plot y = 1/(1 + e^(-2x-1))" and play with the parameters of the linear equation. Notice that the sigmoid function always maintains its basic shape, but you can control the placement of the step, as well as the steepness of the step. The placement of the step marks the decision boundary between the classes. The steepness of the step, I *assume*, affects the confidence of the classification for the values near the decision boundary.

Alex Holehouse's notes on logistic regression provide some very valuable insight into the meaning of the linear equation term (theta transpose x) in this case.

In this case, the linear equation (theta transpose x) defines a "discriminant" function which gives a positive value for one class and a negative value for another class. If you set this function to 0, you will find the decision boundary. Using a linear function, you can only distinguish classes that are linearly separable (e.g., you can draw a straight line to reasonably separate the two classes).

Finally, it's important to have an intuitive understanding of the probability notation p (y | x). He starts referring to p(y | x) a lot in the next couple lectures.

For logistic regression, we're making the assumption that the sigmoid function that we've fitted to our binary data is also a good measure of the probability.

**![](https://lh5.googleusercontent.com/lRD2dDH99Qs1cHJgjHIJQpzWR37J7CfwfUVFK9wdfxDwopUOc0HmczH1sFeZylXoBhErsYf_-ifL2505NpCUi2q2EKIdyG5VAvamTpOlWt1OOPPbcVdrV0la)**

Which can also be written as:

**![](https://lh6.googleusercontent.com/IMWUNn0q_0Ofk9qEZqueGTKTrqSWgUhfeOjd-yarOMesJpUpJ8anBwjkvsKEJZc6ES39IMRcTo1JYofccCn_dwiputsGoq0EcBId1qTTJ7aQx2K7UM-YL24J)**

The important thing to understand with these is that the only meaningful values for y in this case are 0 or 1. Also, the semi-colon theta can be read as "given the model you've come up with which its theta values that you've found".

He will start writing p(y | x) more often then h(x). Whenever you see p(y | x) in the context of logistic regression, I think a good way to read this is as "the probability that input 'x' belongs to class 'y'". For example, the probability that a tumor of size x is malignant.

If p(y | x) is written in relation to regression (such as fitting a line to house price per square feet), it's a little more complicated. _y_ can take on any value. So for a given square footage x, p(y | x) is a bell curve where the horizontal axis is the all of the possible prices of the house, and the peak of the bell curve will be centered on the price predicted by your hypothesis function. So for the housing price model you've come up with, p(y | x) is the probability that a house with x square feet will be worth y dollars, according to your model.

**Digression Perceptron**

This topic is only briefly mentioned. Instead of using the sigmoid for the hypothesis function, you just use a threshold so that you have a step function. It sounds like we'll come back to this topic later.
