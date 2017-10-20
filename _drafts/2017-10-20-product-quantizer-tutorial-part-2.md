---
layout: post
title:  "Product Quantizers for k-NN Tutorial Part 2"
date:   2017-10-20 7:00:00 -0800
comments: true
image: /assets/ProductQuantizer/compression.png
tags: similarity search, FAISS, k-NN, k-nearest neighbors, k-NN search, product quantizer, approximate nearest neighbors, ANN, quantization
---

* TOC
{:toc}

In the previous post, I described the most basic form of a product quantizer. In the 2011 paper which introduced PQs as a method for k-NN, though, there were a few other techniques that they combined with the basic PQ, and I'll be covering those in this part 2.

Taken all together, these techniques make up the IndexIVFPQ class in FAISS.

Here is a brief summary of the two added features, followed by more detailed explanations.

*Inverted File Index (IVF)*
The IVF is simply a technique for pre-filtering the dataset so that you don't have to do an exhaustive search of _all_ of the vectors. It's pretty straightforward--you cluster the dataset ahead of time with k-means clustering to produce a large number of (e.g., 100) dataset partitions. Then, at query time, you compare your query vector to the partition centroids to find, e.g., the 10 closest clusters, and then you search against only the vectors in those partitions.  

*Encoding Residuals*
This is an enhancement to the basic product quantizer which incorporates information from the IVF step. For each database vector, instead of using the PQ to encode the original database vector we instead encode the vector's _offset_ from its partition centroid.

## Inverted File Index
In Computer Science, and in Information Retrieval in particular, an "inverted index" refers to a text search index which maps every word in the vocabulary to all of its locations in all of the documents in the database. It's a lot like the index you'd find in the back of a textbook, mapping words or concepts to page numbers, so it's always bugged me that they call this data structure an _inverted_ index (cause it seems like a _normal_ index to me!).

Anyhow, in this context, the technique really just means partitioning the dataset using k-means clustering so that you can refine your search to only some of the partitions and ignore the rest.

As part of building the index, you use k-means clustering to cluster the dataset into a large number of partitions. Each vector in the dataset now belongs to one (and only one) of these clusters / partitions. And for each partition, you have a list of all the vectors that it contains (these are the "inverted file lists" referred to by the authors). You also have a matrix of all of the partition centroids, which will be used to figure out which partitions to search.

Dividing the dataset up this way isn't perfect, because if a query vector falls on the outskirts of the closest cluster, then it's nearest neighbors are likely sitting in multiple nearby clusters. The solution is simply to search multiple partitions.

At search time, you compare your query vector to all of the partition centroids to find the closest ones. Exactly how many is configurable (TODO - What's the default ratio between nlist and nprobe?). Once you've found the closest centroids, you select only the dataset vectors from those partitions, and do your k-NN search using the product quantizer.

A few notes on terminology:
* The verb "probe" is used in this context to refer to selecting partitions to search. So in the code you'll see the parameter as "number of partitions to probe".
* The authors of FAISS like to use the term Voronoi cells. A Voronoi cell is just the region of space that belongs to a cluster. That is, it covers all the points in space where a vector would be closer to that cluster's centroid than any other.

## Encoding Residuals
This step is also relatively straightforward, but it convolutes the explanation of the index a good deal! The idea is to incorporate some of the information from the IVF step into the product quantizer to improve its accuracy (so this concept definitely builds _on top of_ the dataset partitioning technique).

First let's define what a 'residual' vector is. And let's tear out the product quantizer for now since it complicates things--we'll come back to it later.

Let's say you've clustered your dataset with k-means and now you have 100 clusters / dataset partitions. For a given dataset vector, its residual is its offset from its partition centroid. That is, take the dataset vector and subtract from it the centroid vector of the cluster it belongs to.

If you replace all of the vectors in a cluster with their "residual", you can think of this as defining a new coordinate system, where the cluster centroid is now at (0, 0, 0, ...) and all of the cluster members now have their coordinates defined relative to that centroid.

So here's something interesting. Let's say you replace all of the vectors in a partition with their residuals. If you now have a query vector, and you want to find its nearest neighbors within this partition, you can calculate the query vector's residual (it's offset from the partition centroid), and then do your nearest neighbor search against the dataset vector residuals. And you'll get the same result as using the original vectors!

If you picture a 2D example in your head, I think you can see this intuitively. But let's also look at the equation.

d(x, y) = d(x - c, y - c)

sqrt(sum_i ((x_i - c_i) - (y_i - c_i))^2)

The centroid components just cancel out.

So note that this means that the distances calculated using the residuals are not only equivalent in relative terms (like the order of the distances), but we're actually still calculating the correct L2 distance between the vectors!

You've probably already relied on this equivalence before, since mean normalization (subtracting the mean from your vectors) is a common pre-processing technique.

So that's within a single partition, but what about comparing against vectors in different partitions? We're still good! That is, even when working with residuals, you can safely compare the distances calculated in one partition with the distances calculated in another partition, because they're still all the correct L2 distance values!

To state this in equation form, we want to know if the distance between our query vector x with vector a in partition 1 is less than the distance to vector b in partition 2.

d(x, a) < d(x, b)

And this is equivalent to

d(x - c_1, a - c_1) < d(x - c_2, b - c_2)

The only thing to note here is that for each partition we probe, we do have to re-calculate the residual for the query vector based on that partition's centroid.

So this is all very interesting, but so far useless. It's not buying us anything.

Let's bring the PQ back in to the picture. Before training our Product Quantizer, we're going to replace all of the dataset vectors with their residuals. So now our product quantizer works on residuals instead of on the original vectors.

I believe what this buys us is that, by replacing all of the vectors with their offset from a nearby centroid, we've reduced the variety in the dataset. In the paper, this is described by saying that the residual vectors "have less energy" than the originals. Our limited number of codes in the PQ are now going to be more accurate because the vectors they have to describe are less distinct than they were. We're getting more bang for our buck!

There is a cost to this, though. Remember that the magic of Product Quantizers is that you only have to calculate a relatively small table of partial distances between the subvectors in the query vector and the subvector centroids. The rest is look-ups and summations.

But now with the residuals, the query vector is different for each partition--in each partition, the residual for the query has to be re-calculated against that partition's centroid. So we have to calculate a separate distance table for each partition we probe!

Apparently the trade-off is worth it, though, because the IndexIVFPQ works pretty well in practice.

And that's really it. The database vectors have all been replaced by their residuals, but from the perspective of the Product Quantizer nothing's different.

You could train separate PQs for each partition, but the authors decided against this because the number of partitions tends to be high, so the memory cost of storing all those codebooks is a problem. Instead, a single PQ is still learned from all of the database vectors from all partitions.



7:00p - 7:10p

TODO - Do they re-organize the vectors so that the vectors in a partition are all contiguous on disk or in memory?
