---
author: chrisjmccormick
comments: true
date: 2014-07-09 19:51:45+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5915
published: false
slug: spherical-k-means-clustering
title: Spherical K-Means Clustering
wordpress_id: 5915
---

Spherical k-means


### Cosine Similarity


Standard k-means uses the Euclidean distance between two vectors as a similarity metric for clustering. Two vectors which are separated by a smaller Euclidean distance are considered more alike than two vectors with a larger distance between them. This is used in the cluster assignment step of standard k-means--we assign a point to a cluster based on the Euclidean distance between the point and the cluster center.

In spherical k-means, we will be using "cosine similarity" as the similarity metric in place of the Euclidean distance.

From the [Wikipedia article on cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity):


<blockquote>"**Cosine similarity** is a measure of similarity between two vectors...that measures the [cosine](http://en.wikipedia.org/wiki/Cosine) of the angle between them. The cosine of 0° is 1, and it is less than 1 for any other angle. It is thus a judgement of orientation and not magnitude: two vectors with the same orientation have a Cosine similarity of 1, two vectors at 90° have a similarity of 0, and two vectors diametrically opposed have a similarity of -1, independent of their magnitude"</blockquote>


We won't be directly calculating the angle and its cosine, however. Instead, we take the normalized dot product between the two vectors:

[![CosineSimilarityEq](http://chrisjmccormick.files.wordpress.com/2014/07/cosinesimilarityeq.png?w=470)](https://chrisjmccormick.files.wordpress.com/2014/07/cosinesimilarityeq.png)




### 


Recall that standard k-means consists of two simple steps which repeat over and over until the clusters stop changing:



	
  1. Assign all points to the closest cluster.

	
  2. Move the center of each cluster to be the mean (the "centroid") of all the cluster members.


Spherical k-means has the same steps, except that we will be using the cosine similarity instead of Euclidean distance to perform the assignment in step 1.


