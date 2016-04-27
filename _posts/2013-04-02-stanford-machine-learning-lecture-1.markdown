---
author: chrisjmccormick
comments: true
date: 2013-04-02 17:23:59 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/04/02/stanford-machine-learning-lecture-1/
slug: stanford-machine-learning-lecture-1
title: Stanford Machine Learning - Lecture 1
wordpress_id: 5490
tags:
- Course Overview
- Lecture Notes
- Machine Learning
- Stanford
---

This first lecture starts out with some logistics for the class. The real material starts at about [31:40], but I'd recommend watching it all anyway because he does give you a feel for his goals for the class, as well as mention some applications of machine learning in use today. The material for this lecture is just to provide you an overview of the major topics that the course will cover.

Current applications:



	
  * Character recognition requires a machine learning algorithm.

	
  * 


Data mining - One example is the digital database of medical records.




	
  * 




US Postal Services uses machine learning algorithm to recognize the zipcode you wrote on your check.

	
  * 


Many of your checks are processed by a machine learning algorithm to recognize the dollar amount.




	
  * 




Credit card transactions - learning algorithm trying to figure out if your card has been stolen, if this is a fraudulent transaction

	
  * 


Netflix - video recommendations.




	
  * Additional applications mentioned in the Coursera course:

	
    * Google / Bing - Machine learning algorithm decides what search results to show

	
    * Facebook / Apple's photo tagging features

	
    * Spam filtering

	
    * Autonomous robotics

	
    * Computational biology

	
    * Data mining

	
      * Web click data / click stream data

	
      * Medical records

	
      * Biology

	
      * Engineering




	
    * Applications that can't be programmed by hand:

	
      * Hand writing recognition

	
      * Natural language processing (NLP)

	
      * Computer Vision




	
    * Product recommendations, e.g. Amazon, Netflix, iTunes





Pre-requisites for the course:

	
  * Basic programming

	
    * The course is mostly programming in Matlab and Octave (Octave is freely available).




	
  * Basic statistics

	
  * Linear algebra

	
  * You can find review materials for the prerequisites on the handouts page here:

	
    * [http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx](http://see.stanford.edu/see/materials/aimlcs229/handouts.aspx)




	
  * He says the discussion sections (which cover the review material) are also televised, but I think they must have chosen not to distribute those videos (I can't find them anywhere).




What is machine learning?






	
  * Arthur Samuel (1959): Field of study that gives computers the ability to learn without being explicitly programmed.




Supervised Learning






	
  * Provide the algorithm a data set (houses with square footage and their listing price)

	
  * You’re giving the algorithm the “right answer”, having it learn the relationship between input and output.




Regression






	
  * Involves a continuous variable which you are trying to predict (e.g., housing price based on square footage of home).




Classification






	
  * The variable you’re trying to predict is discrete, not continuous.

	
  * Example, predict whether a tumor is malignant. To make it simple, just look at tumor size.

	
  * The y-axis is just 0 or 1 (malignant or non-malignant).

	
  * Or, you could have multiple variables (age and tumor size), and use different symbols for representing malignant or benign.




Learning Theory






	
  * Helps us answer the question--do I have enough training data to feel confident in the accuracy of my system?

	
  * In this course, he aspires to help you be able to apply machine learning really well. This is often missing from other courses.


Unsupervised Learning

	
  * You don’t have the right answers for the data. Try to find interesting structure in the data.

	
  * Clustering is one example of unsupervised learning.

	
  * He gives an example of a PhD student working on clustering gene data.

	
  * Example of clustering pixels in an image to divide the image into regions.

	
  * Applications of clustering:

	
    * Organizing computer clusters

	
    * Social network analysis - Can we organize friends into circles or groups based on interactions?

	
    * Market segmentation





****
![](https://lh4.googleusercontent.com/Pz6SY1UM4R7m7W5YrvccRHtzXqbB7YSaLTEIDEEUeFoePwNf4Z17xpjWDbmFEaF6VXPPYPha0ZlUyIZJ6vjLHkfG-yT4ixQra-C_yVyWe02WPN3RjpaSx3hl)
****



	
  * He shows the “Cocktail problem” as another example of unsupervised learning--two people speaking at the same time, how do you separate their voices? ICA - Independent Component Analysis.

	
  * Other examples from the Coursera course:

	
    * Google news groups similar stories (stories about the same event)







Reinforcement Learning






	
  * Autonomous helicopter - If you make a bad decision in controlling it, you won’t crash until later. You need to make a sequence of bad decisions.

	
  * Basic idea is to have a reward mechanism. Every time it does something right, you reinforce that, when it makes a mistake, you minimize it.

	
  * Seems to be a lot of application to robotics--four legged dog robot, snake robot, self driving RC car, obstacle climbing bot.


