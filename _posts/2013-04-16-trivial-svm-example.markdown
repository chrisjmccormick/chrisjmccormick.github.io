---
author: chrisjmccormick
comments: true
date: 2013-04-16 21:42:40 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/04/16/trivial-svm-example/
slug: trivial-svm-example
title: SVM Tutorial - Part I
wordpress_id: 5587
tags:
- Kernel
- Support Vector Machines
- Support Vectors
- SVM
- SVM Example
- WEKA
---

I found it really hard to get a basic understanding of Support Vector Machines. To learn how SVMs work, I ultimately went through Andrew Ng's Machine Learning course (available freely from Stanford).  I think the reason SVM tutorials are so challenging is that training an SVM is a complex optimization problem, which requires a lot of math and theory to explain.

However, just by looking at an SVM that's been trained on a simple data set, I think you can gain some of the most important insights into how SVMs work.

I fabricated a very simple SVM example in order to help myself understand some of the concepts. I've included the results and illustrations here, as well as the data files so that you can run the algorithm yourself.

In Excel, I created two sets of points (two "classes") that I placed arbitrarily. I placed the points such that you can easily draw a straight line to separate the two classes (the classes are "linearly separable"). These points are my training set which I used to train the SVM.

[![TrivialDataset](http://chrisjmccormick.files.wordpress.com/2013/04/trivialdataset.png)](http://chrisjmccormick.files.wordpress.com/2013/04/trivialdataset.png)

Here is the [Excel spreadsheet](http://chrisjmccormick.files.wordpress.com/2013/04/supportvectormachines.xlsx) containing the data values and the plots in this post.


## SVM Scoring Function


A trained Support Vector Machine has a scoring function which computes a score for a new input. A Support Vector Machine is a binary (two class) classifier; if the output of the scoring function is negative then the input is classified as belonging to class y = -1. If the score is positive, the input is classified as belonging to class y = 1.

Let's look at the equation for the scoring function, used to compute the score for an input vector _x._

[![ScoringFunction](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunction.png)](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunction.png)






	
  * This function operates over every data point in a training set (_i_ = 1 through _m_).

	
    * Where _x_^(i), _y_^(i) represents the _i_th training example. (Don't confuse this as "x to the ith power")

	
    * _x_^(i) is an input vector which may be any dimension.

	
    * _y^_(i) is a class label, which has one of only two values, either -1 or 1.

	
    * α_i is the coefficient associated with the _i_th training example.




	
  * _x_ is the input vector that we are trying to classify

	
  * _K_ is what is called a kernel function.

	
    * It operates on two vectors and the output is a scalar.

	
    * There are different possible choices of kernel function, we'll look at this more later.




	
  * _b_ is just a scalar value.




## Support Vectors


This scoring function looks really expensive to compute. You'll have to perform an operation on every single training point just to classify a new input _x_--what if your training set contains millions of data points?  As it turns out, the coefficient α_i will be zero for all of the training points except for the "support vectors".









In the below plot, you can see the support vectors chosen by the SVM--the three training points closest to the decision boundary.







[![TrivialDataset_SupportVectors](http://chrisjmccormick.files.wordpress.com/2013/04/trivialdataset_supportvectors.png)](http://chrisjmccormick.files.wordpress.com/2013/04/trivialdataset_supportvectors.png)







## Training The SVM In WEKA




To train an SVM on this data set, I used the freely available [WEKA toolset](http://www.cs.waikato.ac.nz/ml/weka/).







1. In the WEKA explorer, on the 'Preprocess' tab, open [this .csv file](https://docs.google.com/file/d/0B-kWgXJRQkQ7VGszZTJvYjc1Z0k/edit?usp=sharing) containing the data set.







[![OpenFile](http://chrisjmccormick.files.wordpress.com/2013/04/openfile.png)](http://chrisjmccormick.files.wordpress.com/2013/04/openfile.png)







2. On the 'Classify' tab, press the "Choose" button to select classifier weka->classifiers->functions->SMO (SMO is an optimization algorithm used to train an SVM on a data set).







[![ChooseSMO](http://chrisjmccormick.files.wordpress.com/2013/04/choosesmo.png)](http://chrisjmccormick.files.wordpress.com/2013/04/choosesmo.png)






3. Click on the classifier command line to bring up a dialog for editing the command line arguments.






In the dialog, change the 'filterType' property to "No normalization/standardization". This will make the results easier to interpret.




Also, click the command line of the 'kernel' property. This will bring up another dialog to allow you to specify properties of the kernel function. Set the 'exponent' property to 2.







[![SetupPolyKernel](http://chrisjmccormick.files.wordpress.com/2013/04/setuppolykernel.png)](http://chrisjmccormick.files.wordpress.com/2013/04/setuppolykernel.png)







A note for those who are already familiar with kernels: Since our data set is linearly separable, we don't really need an exponent of 2 on the kernel. This is necessary, though, to force WEKA to use support vectors.  Otherwise, it will just give you a simple linear equation for the scoring function, which doesn't help us in our understanding of SVMs.







4. Click 'Start' to run the training algorithm.







Towards the middle of the output, you should see something like the following equation:









    
        0.0005 * <7 2 > * X]
     -  0.0006 * <9 5 > * X]
     +  0.0001 * <4 7 > * X]
     +  2.7035







This is essentially the scoring function that we saw at the beginning of the post, but now with the values filled in.







The numbers in angle brackets are our three support vectors <7  2>, <9  5>, and <4  7> (these are the points I marked in the scatter plot).







The coefficient beside each support vector is the computed 'alpha' value for that data point.







The sign of the coefficient comes from the class label. For example, <9  5> belongs to class y = -1, and <7  2> belongs to class y = 1. In the original scoring function, there was the term α_i * _y_^(i). The alpha values will always be greater than or equal to 0, so the sign of the coefficient comes from the class label _y_^(i).







The final value in the expression, 2.7035, is _b_.







## Visualizing The Scoring Function




Now we can take our scoring equation:




[![ScoringFunction](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunction.png)](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunction.png)







And plug in the values we've found to get:







[![ScoringFunctionValues](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunctionvalues.png)](http://chrisjmccormick.files.wordpress.com/2013/04/scoringfunctionvalues.png)







Where x1 and x2 are the components of the input vector _x _that we want to compute a score for.







Note that the original summation was over every point in the training set, but we've only included the terms for the support vectors here. Alpha is zero for all of the other data points, so those terms disappear.







You can plot the above equation using Google; paste the following into a Google search bar:







plot z = 0.0005*(7x + 2y)^2-0.0006*(9x+5y)^2+0.0001(4x+7y)^2+2.7035







Change the ranges to x: 0 - 16 and y: 0 - 16 and you should get something like the following:







[![Hypothesis](http://chrisjmccormick.files.wordpress.com/2013/04/hypothesis.png)](http://chrisjmccormick.files.wordpress.com/2013/04/hypothesis.png)










The scoring function forms a surface in three dimensions. Where it intersects the z = 0 plane it forms a line; this is our decision boundary. To the left of the decision boundary, inputs receive a score higher than 0 and are assigned to class y = 1. To the right inputs receive a score less than 0 and are assigned to class y = -1.







In the next example, we'll look at a slightly more complex classification problem where the classes are not linearly separable. In that example, we'll go into more detail about the kernel function and how it's used to achieve non-linear classification.
