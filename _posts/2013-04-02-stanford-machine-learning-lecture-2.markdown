---
author: chrisjmccormick
comments: true
date: 2013-04-02 18:12:42 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/04/02/stanford-machine-learning-lecture-2/
slug: stanford-machine-learning-lecture-2
title: Stanford Machine Learning - Lecture 2
wordpress_id: 5492
tags:
- Gradient Descent
- Lecture Notes
- Linear Regression
- Machine Learning
- Normal Equations
- Stanford
---

This lecture covers:



	
  * Linear regression

	
    * Linear regression is the problem of fitting a linear equation to a set of data.




	
  * Gradient descent

	
    * Gradient descent is an iterative approach to fitting a line.




	
  * Normal equations

	
    * Uses the same criteria for line fitting as gradient descent, but does it by explicitly solving the equations.





**Resources**

_Linear Algebra Review_

Before watching this lecture, I read through some of the discussion section notes on Linear Algebra:

[http://see.stanford.edu/materials/aimlcs229/cs229-linalg.pdf](http://see.stanford.edu/materials/aimlcs229/cs229-linalg.pdf)

For this lecture, you just need to be familiar with basic matrix operations like multipying matrices and vectors, taking the transpose, etc. Lecture 2 _doesn't_ go into things like the determinant of a matrix or eigen values.

Also see Holehouse's notes on Linear Algebra below.

_Course Lecture Notes_

There are also lecture notes for this lecture: [http://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf](http://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf) (Available from the [handouts](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx) page of the SEE site).

I recommend going back and forth between the video and the lecture notes. After you watch a portion of the video, read the correspodning section in the lecture notes to ensure you understand it correctly.

This lecture appears to end at section 2.2 in the lecture notes, presumably picking up at section 3 of the lectures notes in the next lecture.

_ Additional Lecture Notes_

Alex Holehouse has put together his own notes on the lectures, including some nice diagrams and examples. Definitely give his notes a look. He goes into more depth in some places than what professor Ng covers in the lectures (unless he's adding in material from later lectures?).

The following two sections of his notes pertain to lecture 2. In section four of his notes, he gets into the discussion of features a bit, which Ng doesn't cover till the beginning of lecture 3, but I think you'll still be able to follow Holehouse's notes.

[http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html](http://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf  I recommend going back and forth between the video and the lecture notes. After you watch a portion of the video, read the correspodning section in the lecture notes to ensure you understand it correctly.)

[http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html](http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html)

He also has his own notes on some linear algebra basics that can help you get through this lecture.

[http://www.holehouse.org/mlclass/03_Linear_algebra_review.html](http://www.holehouse.org/mlclass/03_Linear_algebra_review.html)


## My Notes


[3:20 - 9:15]



	
  * Alvin, car trained to drive by watching human driver, trained using neural network.

	
    * Note that the output was not a category, but a continuous value--the steering direction.




	
  * Alvin is an example of a regression problem, because you are trying to predict a continuous variable, which is the steering direction.


**Batch Gradient Descent**



	
  * If a small increase in theta produces a positive change (increase) in the total error, then reduce theta. And vice versa.

	
    * The amount you modify theta by is the change in error.

	
      * If the a small change in theta produces a small change in error, then change theta by a small amount. Maybe we’re approaching the minima?

	
      * If a large change in theta produces a large change in error, change theta by a large amount. Maybe we have a ways to go to the get to the minima?

	
      * In either case, the change is scaled by the “learning rate”, which you can choose to make the descent more aggressive or conservative.




	
    * In trying to intuitively understand this update rule function, remember that we’re using a single training example (x1, x2, y). So we evaluate this function using just those values for now.




	
  * I haven’t been able to puzzle out why, in the update function, you multiply the error by the data point itself.

	
  * To handle multiple training points, you take the sum of the proposed updates from every data point.

	
  * I believe the contour graph is a simplified 3D graph in which the graph surface is represented by contours (like a topographical map) of different colors to represent their height.

	
  * You do this until it “converges”. Convergence is measured with some heuristics, like “small change in theta over 2 iterations”, or “small change in J”.

	
  * The changes in theta naturally get smaller as you approach the minima:

	
    * As you approach the minima, the error gets smaller and you automatically take smaller and smaller steps.

	
    * You will likely/possibly(?) overstep the minima, but then the sign on the update function will switch, and you will head back towards the minima.





**Stochastic Gradient Descent**



	
  * Just use a single training example and upate all the thetas, then move on to the next training example and repeat for all of them.


**Normal Equations**

[51:00]



	
  * You can also derive the “closed form” solution of theta. That is, you should be able to derive an equation to explicitly compute the theta values.

	
  * To do this by hand, you would have one equation for every training example. The thetas are your unknowns.


I didn’t follow the whole derivation, but I did try to make sure I understood the result.

Here is an explanation of the effect of each operation in the result.
<table cellpadding="0" cellspacing="0" border="0" >
<tbody >
<tr >

<td valign="top" >_X_
</td>

<td valign="top" >_X_ is a matrix containing all of the training example input values. Each row is a different training example, and there is one column for each input variable (x1 , x2 , etc..)
</td>
</tr>
<tr >

<td valign="top" >_X_T_X_
</td>

<td valign="top" >If you multiply a matrix by its transpose, the result is a square matrix. A matrix must be square before you can invert it.
</td>
</tr>
<tr >

<td valign="top" >(_X_T_X_)-1
</td>

<td valign="top" >_X_T_X _gives a square matrix, and the inverse of that is another square matrix. I’m not familiar with the procedure for computing the inverse of a matrix, but there is one.
</td>
</tr>
<tr >

<td valign="top" >(_X_T_X_)-1_X_T
</td>

<td valign="top" >Multiplying the previous step by _X_T yields another square matrix.
</td>
</tr>
<tr >

<td valign="top" >(_X_T_X_)-1_X_T_y_
</td>

<td valign="top" >Multiply this by the vector y and you get another vector. This final vector is the list of theta values.
</td>
</tr>
</tbody>
</table>
The key trade-off here (vs. gradient descent) is the computation of the inverse matrix. The compute time for this step grows exponentially with the size of the matrix, so it may not be feasible for very large data sets.
