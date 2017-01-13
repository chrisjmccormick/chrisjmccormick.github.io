---
layout: post
title:  "DBSCAN Clustering"
date:   2016-11-08 22:00:00 -0800
comments: true
image: assets/dbscan/demo_screenshot.png
tags: DBSCAN, Clustering, Density-Based Clustering
---

DBSCAN is a popular clustering algorithm which is fundamentally very different from k-means. 

* In k-means clustering, each cluster is represented by a centroid, and points are assigned to whichever centroid they are closest to. In DBSCAN, there are no centroids, and clusters are formed by linking nearby points to one another. 
* k-means requires specifying the number of clusters, 'k'. DBSCAN does not, but does require specifying two parameters which influence the decision of whether two nearby points should be linked into the same cluster. These two parameters are a distance threshold, \\( \varepsilon \\) (epsilon), and "MinPts" (minimum number of points), to be explained. 
* k-means runs over many iterations to converge on a good set of clusters, and cluster assignments can change on each iteration. DBSCAN makes only a single pass through the data, and once a point has been assigned to a particular cluster, it never changes.

I like the language of trees for describing cluster growth in DBSCAN. It starts with an arbitrary seed point which has at least MinPts points nearby within a distance (or "radius") of \\( \varepsilon \\). We do a breadth-first search along each of these nearby points. For a given nearby point, we check how many points *it* has within its radius. If it has fewer than MinPts neighbors, this point becomes a *leaf*--we don't continue to grow the cluster from it. If it *does* have at least MinPts, however, then it's a *branch*, and we add all of its neighbors to the FIFO queue of our breadth-first search.

Once the breadth-first search is complete, we're done with that cluster, and we never revisit any of the points in it. We pick a new arbitrary seed point (which isn't already part of another cluster), and grow the next cluster. This continues until all of the points have been assigned.

There is one other novel aspect of DBSCAN which affects the algorithm. If a point has fewer than MinPts neighbors, *AND it's not a leaf node of another cluster*, then it's labeled as a "Noise" point that doesn't belong to any cluster. 

Noise points are identified as part of the process of selecting a new seed--if a particular seed point doesn't have enough neighbors, it's labeled as a Noise point. This label is often temporary, however--these Noise points are often picked up by some cluster as a leaf node. 

Visualization
=============
Naftali Harris has created a great web-based [visualization](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) of running DBSCAN on a 2-dimensional dataset. Try clicking on the "Smiley" dataset and hitting the GO button. (That's where the image from this post came from). Very cool!

Algorithm in Python
===================
To fully understand the algorithm, I think it's best to just look at some code.

Below is a working implementation in Python. Note that the emphasis in this implementation is on illustrating the algorithm... the distance calculations, for example, could be optimized significantly.

You can also find this code (along with an example that validates it's correctness) on GitHub [here](https://github.com/chrisjmccormick/dbscan).

{% highlight py %}

import numpy

def MyDBSCAN(D, eps, MinPts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(D)

    # C is the ID of the current cluster.    
    C = 0
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, len(D)):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        NeighborPts = regionQuery(D, P, eps)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           # Get the next cluster label.
           C += 1
           
           # Assing the label to our seed point.
           labels[P] = C
           
           # Grow the cluster from the seed point.
           growCluster(D, labels, P, C, eps, MinPts)
    
    # All data has been clustered!
    return labels


def growCluster(D, labels, P, C, eps, MinPts):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # SearchQueue is a FIFO queue of points to evaluate. It will only ever 
    # contain points which belong to cluster C (and have already been labeled
    # as such).
    #
    # The points are represented by their index values (not the actual vector).
    #
    # The FIFO queue behavior is accomplished by appending new points to the
    # end of the list, and using a while-loop rather than a for-loop.
    SearchQueue = [P]

    # For each point in the queue:
    #   1. Determine whether it is a branch or a leaf
    #   2. For branch points, add their unclaimed neighbors to the search queue
    i = 0
    while i < len(SearchQueue):    
        
        # Get the next point from the queue.        
        P = SearchQueue[i]

        # Find all the neighbors of P
        NeighborPts = regionQuery(D, P, eps)
        
        # If the number of neighbors is below the minimum, then this is a leaf
        # point and we move to the next point in the queue.
        if len(NeighborPts) < MinPts:
            i += 1
            continue
        
        # Otherwise, we have the minimum number of neighbors, and this is a 
        # branch point.
            
        # For each of the neighbors...
        for Pn in NeighborPts:
           
            # If Pn was labelled NOISE during the seed search, then we
            # know it's not a branch point (it doesn't have enough 
            # neighbors), so make it a leaf point of cluster C and move on.
            if labels[Pn] == -1:
               labels[Pn] = C
            # Otherwise, if Pn isn't already claimed, claim it as part of
            # C and add it to the search queue.   
            elif labels[Pn] == 0:
                # Add Pn to cluster C.
                labels[Pn] = C
                
                # Add Pn to the SearchQueue.
                SearchQueue.append(Pn)
            
        # Advance to the next point in the FIFO queue.
        i += 1        
    
    # We've finished growing cluster C!


def regionQuery(D, P, eps):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors
{% endhighlight %}

