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

I believe what this buys us is that, by replacing all of the vectors with their offset from a nearby centroid, we've reduced the variety in the dataset. In the paper, this is described by saying that the residual vectors "have less energy" than the original.  Our limited number of codes in the PQ are now going to be more accurate because the vectors they have to describe are less distinct than they were. We're getting more bang for our buck!

There is a cost to this, though. The number of partial distance tables we have to calculate has grown.


 Let's talk first about residuals  To calculate the residual for a given database vector, you just subtract it's centroid from it. Think

While playing with FAISS, I used a dataset of 50,000 feature vectors representing the MNIST training set. I

TODO - Do they re-organize the vectors so that the vectors in a partition are all contiguous on disk or in memory?








A product quantizer is a type of “vector quantizer” (I’ll explain what that means later on!) which can be used to accelerate approximate nearest neighbor search. They’re of particular interest because they are a key element of the popular [Facebook AI Similarity Search (FAISS) library](https://code.facebook.com/posts/1373769912645926/faiss-a-library-for-efficient-similarity-search/) released in March 2017. In this post, I’ll be providing an explanation of a product quantizer in its most basic form, as used for implementing approximate nearest neighbors search (ANN).

## Exhaustive Search with Approximate Distances
Unlike tree-based indexes used for ANN, a k-NN search with a product quantizer still performs an “exhaustive search”, meaning that a product quantizer still requires comparing the query vector to every vector in the database. The key is that it _approximates_ and _greatly simplifies_ the distance calculations.

However, it is possible to combine a product-quantizer with different pre-filtering techniques to reduce the number of comparisons performed. The FAISS library includes modes which combine the PQ approach with a pre-filtering step that isolate the search to just a portion of the overall database. I hope to write another tutorial that will cover the pre-filtering in FAISS, but in this post, I'll be focusing on just the product quantizer.

## Explanation by Example
The authors of the product quantizer approach have a background in signal processing and compression techniques, so their language and terminology probably feels foreign if your focus is machine learning. Fortunately, if you’re familiar with k-means clustering (and we dispense with all of the compression nomenclature!) you can understand the basics of product quantizers easily with an example. Afterwards, we'll come back and look at the compression terminology.

## Dataset Compression
Let’s say you have a collection of 50,000 images, and you've already performed some feature extraction with a convolutional neural network, and now you have a dataset of 50,000 feature vectors with 1,024 components each.

![Image Vector Dataset][image_vectors]

The first thing we’re going to do is compress our dataset. The number of vectors will stay the same, but we'll reduce the amount of storage required for each vector. Note that what we're going to do is _not the same_ as "dimensionality reduction"! This is because the compressed vectors can’t be compared to one another directly--this will become clear as we go further.

Two important benefits to compressing the dataset are that (1) memory access times are generally the limiting factor on processing speed, and (2) sheer memory capacity can be a problem for big datasets.

Here’s how the compression works. For our example we’re going to chop up the vectors into 8 sub-vectors, each of length 128 (8 sub vectors x 128 components = 1,024 components). This divides our dataset into 8 matrices that are [50K x 128] each.

![Vectors sliced into subvectors][vector_slices]

We’re going to run k-means clustering separately on each of these 8 matrices with k = 256. Now for each of the 8 subsections of the vector we have a set of 256 centroids--we have 8 groups of 256 centroids each.

![K-Means clustering run on subvectors][kmeans_clustering]

These centroids are like “prototypes”. They represent the most commonly occurring patterns in the dataset sub-vectors.

We’re going to use these centroids to compress our 1 million vector dataset. Effectively, we’re going to replace each subregion of a vector with the closest matching centroid, giving us a vector that’s different from the original, but hopefully still close.

Doing this allows us to store the vectors much more efficiently—instead of storing the original floating point values, we’re just going to store cluster ids. For each subvector, we find the closest centroid, and store the id of that centroid.

Each vector is going to be replaced by a sequence of 8 centroid ids. I think you can guess how we pick the centroid ids--you take each subvector, find the closest centroid, and replace it with that centroid’s id.

Note that we learn a _different set of centroids_ for each subsection. And when we replace a subvector with the id of the closest centroid, we are only comparing against the 256 centroids for _that subsection_ of the vector.

Because there are only 256 centroids, we only need 8-bits to store a centroid id. Each vector, which initially was a vector of 1,024 32-bit floats (4,096 bytes) is now a sequence of eight 8-bit integers (8 bytes total per vector!).  

![Compressed vector representation][compression]

## Nearest Neighbor Search
Great. We’ve compressed the vectors, but now you can’t calculate L2 distance directly on the compressed vectors--the distance between centroid ids is arbitrary and meaningless! (This is what differentiates compression from dimensionality reduction).

Here’s how we perform a nearest neighbor search. It’s still going to be an exhaustive search (we’re still going to calculate a distance against all of the vectors and then sort the distances) but we’re going to be able to calculate the distances much more efficiently using just table look-ups and some addition.

Let’s say we have a query vector and we want to find its nearest neighbors.

One way to do this (that isn't so smart) would be to decompress the dataset vectors, and then calculate the L2 distances. That is, reconstruct the vectors by concatenating the different centroids. We're effectively going to do this, but in a much more computationally efficient way than actually decompressing the vectors.

First, we’re going to calculate the squared L2 distance between each subsection of our vector and each of the 256 centroids for that subsection.

This means building a table of subvector distances with 256 rows (one for each centroid) and 8 columns (one foreach subsection). How much effort is it to build this table? If you think about it, this requires the same number of math operations as computing L2 distances between our query vector and 256 dataset vectors.

Once we have this table, we can start calculating approximate distance values for each of the 50K database vectors.

Remember that each database vector is now just a sequence of 8 centroid ids. To calculate the approximate  distance between a given database vector and the query vector, we just use those centroid id’s to lookup the partial distances in the table, and sum those up!

Does it really work to just sum up those partial values? Yes! Remember that we’re just working with squared L2 distances, meaning no square root operation. Squared L2 is calculated by summing up all of squared differences between each component, so it doesn't matter what order you perform those additions in.

So this table approach gives us the same result as calculating distances against the decompressed vectors, but with much lower compute cost.

The final step is the same as an ordinary nearest neighbor search—we sort the distances to find the smallest distances; these are the nearest neighbors. And that's it!

## Compression Terminology
Now that you understand how PQs work, it’s easy to go back and learn the terminology.

A quantizer, in the broadest sense, is something that reduces the number of possible values that a variable has. A good example would be building a lookup table to reduce the number of colors in an image. Find the most common 256 colors, and put them in a table mapping a 24-bit RGB color value down to an 8-bit integer.

When we took the first 128 values of our database vectors (the first of the 8 subsections) and clustered them to learn 256 centroids, these 256 centroids form what’s refered to as a “codebook”. Each centroid (a floating point vector with 128 components) is called a “code”.

Since these centroids are what’s used to represent the database vectors, the codes are also referred to as "reproduction values” or “reconstruction values". You can reconstruct a database vector from its sequence of centroid if ids by concatenating the corresponding codes (centroids).

Since we ran k-means separately on each of the 8 subsections, we actually created eight separate code books.

With these 8 codebooks, though, we can combine the codes to create 256^8 possible vectors! So, in effect, we've created one very large codebook with 256^8 codes.

### Vector quantizers
We have been looking at the Product Quantizer specifically, but there is an even simpler notion of a "vector quantizer". Here is the compression-language definition: A "vector quantizer" takes a vector and "encodes" it by returning the index of a code.

Here is the definition that will probably make more sense to an ML researcher. You cluster your dataset (the _full length_ vectors, no slicing here) with k-means clustering (this is "training the quantizer"). You replace each vector with the id of the cluster it's closest to (this is "encoding the vectors" or "quantizing the vectors").

In general, when you're reading any of the documentation or code around the FAISS library, when you read quantizer just think k-means clustering!

[image_vectors]: {{ site.url }}/assets/ProductQuantizer/image_vectors.png
[vector_slices]: {{ site.url }}/assets/ProductQuantizer/vector_slice.png
[kmeans_clustering]: {{ site.url }}/assets/ProductQuantizer/kmeans_clustering.png
[compression]: {{ site.url }}/assets/ProductQuantizer/compression.png
