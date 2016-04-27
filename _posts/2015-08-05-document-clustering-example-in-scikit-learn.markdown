---
author: chrisjmccormick
comments: true
date: 2015-08-05 23:13:40 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2015/08/05/document-clustering-example-in-scikit-learn/
slug: document-clustering-example-in-scikit-learn
title: Document Clustering Example in SciKit-Learn
wordpress_id: 6037
tags:
- 20 Newsgroups
- Document Clustering
- Feature Hashing
- Latent Semantic Analysis
- LSA
- Python
- scikit-learn
- SVD
- Text Clustering
- tf-idf
- V Measure
---

I've spent some time playing with the [document clustering example](http://scikit-learn.org/stable/auto_examples/document_clustering.html) in scikit-learn and I thought I'd share some of my results and insights here for anyone interested.


## Installation


I found that a good way to get started with scikit-learn on Windows was to install [Python(x, y)](https://code.google.com/p/pythonxy/wiki/Downloads), a bundled distribution of Python that comes with lots of useful libraries for scientific computing. During the the installation, it lets you select which components to install--I'd recommend simply doing the 'complete' installation. Otherwise, make sure to check scikit-learn.

One thing it comes with that I've liked is the Spyder IDE. Spyder feels a lot like the Matlab IDE, which I'm a fan of, and integrates a code editor, console, and variable browser.


## Running the Example


This example has a number of command line options, but you can run it as-is without setting any of them.

The example should run fast--it only takes a few seconds to complete with the default parameters.


## Dataset


The data used comes from the "[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)" dataset. Newsgroups were the original discussion forums, and this dataset contains posts from 20 different topics:
<table border="1" >
<tbody >
<tr >

<td >comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
</td>

<td >rec.autos
rec.motorcycles
rec.sport.baseball
rec.sport.hockey
</td>

<td >sci.crypt
sci.electronics
sci.med
sci.space
</td>
</tr>
<tr >

<td >misc.forsale
</td>

<td >talk.politics.misc
talk.politics.guns
talk.politics.mideast
</td>

<td style="text-align:left;" >talk.religion.misc
alt.atheism
soc.religion.christian
</td>
</tr>
</tbody>
</table>
By default, this example just selects four of the categories ('alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space') to cluster. There are a total of 3,387 entries across these four categories.

If you use the full dataset (all 20 topics), there are a total of 18,846 entries.


## Process & Algorithms




### Vectorization - tf-idf


The text data needs to be turned into numerical vectors. This is done with an object labeled the 'vectorizer' in the code. The default vectorizer method is the [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) approach. For each document, it will produce a vector with 10,000 components (10,000 is the default number, this can be modified with a command line option).

The TfidfVectorizer object has a number of interesting properties.

It will strip all English "stop words" from the document. Stop words are really common words that don't contribute to the meaning of the document. There are actually many of these words--take a quick look [here](http://xpo6.com/list-of-english-stop-words/) for some examples.

It will also filter out terms that occur in more than half of the documents (max_df=0.5) and terms that occur in only one document (min_df=2).

To enforce the maximum vector length of 10,000, it will sort the terms by the number of times they occur across the corpus, and only keep the 10,000 words with the highest counts.

Finally, the vectorizer normalizes each vector to have an L2 norm of 1.0. This is important for normalizing the effect of document length on the tf-idf values. An interesting fact is that if you normalize the vectors (as the example does), then comparing the L2 distances is equivalent to using the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to compare the vectors.


### HashingVectorizer


The code can optionally use the HashingVectorizer instead. The HashingVectorizer is faster, but speed doesn't seem to be a real concern here.

The HashingVectorizer still just counts the terms, but it does this more efficiently by using [feature hashing](https://en.wikipedia.org/wiki/Feature_hashing). Instead of using a hash map to hash words to buckets which contain the word's index in the term vector (word -> bucket -> vector index), you hash the word directly to a vector index (word -> vector index). This means you don't have to build a hash table, but it carries the risk of hash collisions. The risk of two important terms colliding to the same index is low, though, so this trick works well in practice.


### LSA


The example includes optional dimensionality reduction using "Latent Semantic Analysis" (LSA). This is really just using Singular Value Decomposition (SVD), and it's called LSA in the context of text data. It's referred to as "Truncated SVD" because we're only projecting onto a portion of the vectors in order to reduce the dimensionality.

If you're familiar with dimensionality reduction using Principal Component Analysis (PCA), this is also the same thing! My understanding of PCA vs. SVD is that they both arrive at the principal components, but SVD has some advantages in how it's calculated, so it's used more often in practice.

Try using LSA by passing the command line flag "--lsa=256" to reduce the vectors down to 256 components each. Not only does the clustering run faster, but you'll find that the accuracy increases significantly!

LSA can be thought of as a kind of feature extraction. In this case we are identifying the top 256 features, and eliminating the rest. Eliminating the less discriminative features can improve the quality of the distance calculation as a metric of similarity, since it's not incorporating the difference between unimportant features.


### Clustering


Clustering is performed either using the standard k-means clustering algorithm, or a modified version referred to as "Mini-Batch K-Means".

You can read more about Mini-Batch K-Means in the original paper from Google [here](http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf), it's only 2 pages. Basically, it performs iterations using a randomly selected subset of the data. By default, the scikit-learn example uses a batch size of 1,000 (which is a little less than a third of the data).

Initialization is done using "k-means++" by default; this technique is well-described on Wikipedia [here](https://en.wikipedia.org/wiki/K-means%2B%2B). Essentially, the initial cluster centers are still taken from the data, but are chosen so that they are spread out.


## Results


A number of metrics are provided for assessing the quality of the resulting clusters.

Homogeneity, Completeness, and the V-Measure scores are all related. All three of these range from 0 to 1.0, with 1.0 being a perfect match with the ground truth labels. Homogeneity measures the degree to which the clusters contain only elements of the same class. Completeness measures the degree to which all of the elements belonging to a certain category are found in a single cluster. You can cheat each of these individually: To cheat on homogeneity, just assign every data point to its own cluster. To cheat on completeness, just group all of the items into a single cluster. So, the V Measure combines the two metrics into a single value so that there's no cheating :).

The Adjusted Rand-Index tells you how it's doing compared to random guessing. Random labeling yields a score of 0, while perfect labeling yields 1.0.

Here are some V-Measure scores I got from trying different parameters:

[![image (3)](https://chrisjmccormick.files.wordpress.com/2015/08/image-3.png)](https://chrisjmccormick.files.wordpress.com/2015/08/image-3.png)



	
  1. "Mini-batch K-Means": tf-idf vectors with 10,000 terms, using mini-batch k-means

	
  2. "Full K-Means": Same as #1, but using full k-means instead of mini-batch

	
  3. "LSA, 256 Components": Using LSA to compress the vectors to 256 components, and using mini-batch k-means.


I averaged these scores over 5 runs; however, the results vary so much from run to run that for an accurate comparison I'd recommend averaging the results over 100 runs.


