---
author: chrisjmccormick
comments: true
date: 2013-08-16 17:50:47 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/08/16/rbf-network-matlab-code/
slug: rbf-network-matlab-code
title: RBF Network MATLAB Code
wordpress_id: 5702
tags:
- Classification
- Example Code
- Machine Learning
- MATLAB
- Neural Networks
- Octave
- RBF Network
- RBFN
---

UPDATE 8/26: There is now example code for both classification and function approximation.

Below is the Octave / MATLAB code which I used in my two part tutorial on [RBF Networks for classification](http://chrisjmccormick.wordpress.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/) and [RBF Networks for function approximation](https://chrisjmccormick.wordpress.com/2015/08/26/rbfn-tutorial-part-ii-function-approximation/).


## **Classification**


For classification, there is 'runRBFNExample.m', and the example dataset in 'dataset.csv'. Just run the main script and it will load the dataset, train the RBFN, and generate the plots I included in the tutorial.

The dataset came from one of the problem assignments in Andrew Ng's [Machine Learning course](http://www.coursera.org/course/ml) on Coursera. I highly recommend his class if you're at all considering it. The k-means code is also based on the k-means clustering assignment from that class.

The package contains two subdirectories, 'RBFN' and 'kMeans' containing functions specific to those algorithms. The main function will add these two subdirectories to the path for you.

**Using Your Own Dataset**

This code can easily be applied to your own dataset. The key functions are:



	
  * trainRBFN - Train an RBFN on your training data.

	
  * evaluateRBFN - Evaluate the RBFN on a new input to make a classification decision.


The example script 'runRBFNExample.m' provides an example of how to apply these functions. If you use this script as a starting point for your own data, I suggest removing the "Contour Plots" section of the code, lines 55 - 129. That section is specific to the provided dataset and likely isn't applicable to your data (the plots only work / make sense for 2D data).


## **Function Approximation**


For function approximation, look at 'runRBFNFuncApproxExample.m'. It uses a lot of the same code as the classification RBFN, except is uses 'trainFuncApproxRBFN' for training and 'evaluateFuncApproxRBFN' for applying the RBFN to input data.


## **Example Code**


**Update: Revision 1.4**

[RBFN Example Code][example_2015_08_26]
	
  * Added example code for function approximation.


**Older Versions**



	
  * Version 1.3 - [RBFN Example Code - Version 2014_08_18][example_2014_08_18]

	
    * Fixed the print statements for Matlab users--replaced double quotes with single quotes.

	
    * Replaced gradient descent with the "normal equations" (sometimes referred to as the matrix inverse solution for the weights). This approach is simpler, faster, and guaranteed to yield the optimum weight values.




	
  * Version 1.2 - [RBFN Example Code  - Version 2014_04_08][example_2014_04_08]

	
    * Removed calls to the Octave 'rows' function.

	
    * Removed uses of '+=' operator.

	
    * Replaced the Octave 'fminunc' function with 'fmincg' and provided 'fmincg.m'.






	
  * Version 1.1 - [RBFN Example Code - Version 2014_02_14][example_2014_02_14]
    * A number of people had trouble loading the included dataset.mat file in Matlab, so I replaced it with a .csv file instead.
    * There is now a 'trainRBFN' function which encompasses the RBFN training process.
    * The 'trainRBFN' function is set up to handle any number of categories. The original example code was hardcoded to two categories.
    * It is possible for k-Means to choose cluster centers which end up with no members. It's impossible to calculate a beta value for an empty cluster, so the code now removes empty clusters before moving on to calculate the beta values.

	
  * Version 1.0 -** **[RBFN Example Code - Version 2013_08_16][example_2013_08_16]

[example_2013_08_16]: {{ site.url }}/assets/rbfn/RBFN_Example_v2013_08_16.zip
[example_2014_02_14]: {{ site.url }}/assets/rbfn/RBFN_Example_v2014_02_14.zip
[example_2014_04_08]: {{ site.url }}/assets/rbfn/RBFN_Example_v2014_04_08.zip
[example_2014_08_18]: {{ site.url }}/assets/rbfn/RBFN_Example_v2014_08_18.zip
[example_2015_08_26]: {{ site.url }}/assets/rbfn/RBFN_Example_v2015_08_26.zip
