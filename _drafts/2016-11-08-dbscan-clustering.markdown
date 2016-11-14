---
layout: post
title:  "DBSCAN Clustering"
date:   2016-11-08 22:00:00 -0800
comments: true
image: 
tags: DBSCAN, Clustering, Density-Based Clustering
---

DBSCAN is a popular clustering algorithm which is fundamentally very different from k-means. 

* In k-means clustering, each cluster is represented by a centroid, and points are assigned to whichever centroid they are closest to. In DBSCAN, there are no centroids, and clusters are formed by linking nearby points to one another. 
* k-means requires specifying the number of clusters, 'k'. DBSCAN does not, but does require specifying two parameters which influence the decision of whether two nearby points should be linked into the same cluster. These two parameters are a distance threshold, \\( \varepsilon \\) (epsilon), and "MinPts" (minimum number of points), to be explained. 
* k-means runs over many iterations to converge on a good set of clusters, and cluster assignments can change on each iteration. DBSCAN makes only a single pass through the data, and once a point has been assigned to a particular cluster, it never changes.

I like the language of trees for describing cluster growth in DBSCAN. It starts with an arbitrary seed point which has at least MinPts points nearby within a distance (or "radius") of \\( \varepsilon \\). We do a breadth-first search along each of these nearby points. For a given nearby point, we check how many points *it* has within its radius. If it has fewer than MinPts neighbors, this point becomes a *leaf*--we don't continue to grow the cluster from it. If it has at least MinPts, however, then it's a *branch*, and we add all of its neighbors to the FIFO queue of our breadth-first search.

Once the breadth-first search is complete, we're done with that cluster. We pick a new arbitrary seed point (which isn't already part of another cluster), and grow the next cluster. This continues until all of the points have been assigned.

There is one other novel aspect of DBSCAN which affects the algorithm. If a point has fewer than MinPts neighbors, *AND it's not a leaf node of another cluster*, then it's labeled as a "Noise" point that doesn't belong to any cluster. 

Noise points are identified as part of the process of selecting a new seed--if a particular seed point doesn't have enough neighbors, it's labeled as a Noise point. This label isn't always permanent, however; these Noise points can often be picked up later by some cluster as a leaf node.  

Pseudo Code
===========




Popularity
==========

