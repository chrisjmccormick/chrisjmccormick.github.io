---
layout: post
title:  "k-NN Benchmarks Part I - Wikipedia"
date:   2017-09-08 7:00:00 -0800
comments: true
image: assets/wikipedia/banner.png
tags: gensim, python, wikipedia, corpus, concept search, similarity search, nlp, k-NN, GPU, Tesla K80, Nearist, VSX, Vector Search Accelerator, k-nearest neighbors, k-NN search
---

_This post was written in my role as a researcher at Nearist, and will soon be on the [Nearist website](http://www.nearist.io) as well._

This article is the first in a series comparing different available methods for accelerating large-scale k-Nearest Neighbor searches on high-dimensional vectors (i.e., 100 components or more). The emphasis here is on practicality versus novelty--that is, we’re focusing on solutions which are readily available and can be used in production applications with minimal engineering effort. 

At [Nearist](http://www.nearist.io), we have developed a product specifically for accelerating large-scale k-NN search, and have included its performance in these benchmarks to demonstrate it's strengths. That said, we've tried to give each of the platforms a thorough evaluation, and hopefully you will find our results and experiences informative no matter what platform you choose.

This post currently contains the preliminary results of our experiments, and will be updated in the coming months with more data.

* TOC
{:toc}

## The Benchmark

### Concept Search on Wikipedia
Each of our benchmark articles will look at performance on a different k-NN based task. In this article, we will be doing a k-NN search on roughly 4.2 million topic model vectors generated from all of the articles in English Wikipedia. 

This is an example of a technique known as concept search--you take a document, generate a feature vector from it, and then compare it against the feature vectors for a repository of documents (in this case, Wikipedia articles) to find “conceptually similar” documents. 

For example, if we use the Wikipedia article for _Personal computer_ as our query for a k-NN search, we get back the results _Homebuilt computer_, _Desktop computer_, and _Macintosh 128k_ (the original Macintosh).

Parsing all of Wikipedia and generating feature vectors for the text is a big project in its own right. Fortunately, [Radim Řehůřek](https://radimrehurek.com/) has already done exactly this and shared his implementation in the [gensim](https://radimrehurek.com/gensim/) package in Python. I previously [shared a post](http://mccormickml.com/2017/02/22/concept-search-on-wikipedia/) on our experience with this, and also shared a modified and [commented version of the code](https://github.com/chrisjmccormick/wiki-sim-search).

The 'gensim' package supports multiple topic models for converting text into vectors. For these benchmarks, we chose to use Latent Dirichlet Analysis (LDA) with 128 topics (that is, 128 features in the feature vectors).

### Comparison criteria
Over this series of benchmarks, there are multiple aspects to k-NN performance that we are interested in and will evaluate:

* Latency for a single search
* Throughput for a batch of searches
* Memory load time for the GPU and Nearist hardware.
* Accuracy (as impacted by approximation techniques)
* Engineering effort

This first benchmarking post will focus on just the first two--latency and throughput.

## The Contenders
We evaluated three different k-NN implementations:

* k-NN in scikit-learn on a desktop PC (as a baseline)
* GPU acceleration with a Tesla K80
* A prototype version of Nearist’s Vector Search Accelerator (VSX), which uses a custom hardware architecture.

### Desktop CPU
The first platform was simply a brute-force search using the k-NN implementation in scikit-learn in Python, running on a desktop PC, which has a 4th generation Intel i7 4770 (from around 2013) with 16GB of RAM. This was intended merely to serve as a baseline for comparison of the other two platforms.

### Nvidia Tesla K80 GPU

#### knn-cuda library
There don’t appear to be many publicly available kNN GPU implementations; of those, knn-cuda appears to be the most established option. 

Like the Nearist VSX, knn-cuda uses a brute-force method--that is, it calculates and sorts all of the distance values. It accelerates k-NN by leveraging the GPU’s ability to greatly accelerate the matrix-multiplication step at the center of the distance calculations. It also accelerates the sorting of the results.

A note on engineering effort with knn-cuda: While we were able to perform our experiments with the knn-cuda library as-is, the authors note that it “was written in the context of a Ph.D. thesis” and that it “should be adapted to the problem and to the specifications of the GPU used.” That is to say, knn-cuda doesn’t necessarily provide an off-the-shelf implementation of k-NN that is ready for a production application--some further work by a CUDA engineer is required to make best use of the GPU.

#### Amazon P2 instance
Amazon started offering the P2 instances containing Tesla K80s at the end of 2016. GPU users will note that the K80 is now several generations old, and newer Tesla cards would achieve higher performance in our benchmark. However, we wanted to choose the platform with the lowest barrier to entry, and  Amazon instances are trusted and widely used in production applications. Amazon instances have the added benefit of scalability--the ability to increase or decrease the number of GPUs based on demand.

The Tesla K80 actually includes 2 GPU chips (two NVIDIA GK210s) on a single board. CUDA does not automatically divide the computation across the two GPUs for you, however. In order to leverage both GPUs on the K80, you must divide your task across them and then consolidate the results. For this article, we have chosen to simply use a single GK210. In future articles, we plan to publish results using multiple GK210s and even multiple K80 cards.

### Nearist Vector Search Accelerator (VSX)

The VSX Orion is a rack-mounted server appliance that fits in a 4U space. Inside the server are PCIe cards which contain Nearist’s custom hardware accelerators. 

The Orion hardware architecture takes advantage of the easy parallelism in k-NN and divides the search task across hundreds of Vector Comparison Units (VCUs) contained on the PCIe cards in the server. Each VCU operates on a different chunk of the dataset, and then results are aggregated and consolidated by a CPU.

The VSX appliance is still in development, and today we have a much smaller scale prototype. Our prototype consists of a single PCIe card with only about 1/10th of the VCUs that the production version will have. Even this limited prototype outperformed both the CPU and GPU in our benchmark, however.

## Benchmark Results

### Latency Measurements
The following table shows the time taken (in seconds) to perform a k-NN search against the 4.2 million wikipedia articles with only a single query vector. The measurements are all averaged over 5 runs.

Here again is a summary of the platforms:
* Scikit-learn - Python library running on an Intel i7 4770
* GPU - A single GK210 GPU in a Tesla K80
* Nearist - A 1/10th scale prototype Nearist board

These results are preliminary, and we will update them in the coming months. Note that in these current results we are only using half of the compute power on the K80 (since we are only using one GK210 chip), and that the prototype Nearist board only has about 1/10th of the compute power of the production version.

_Latency in seconds_
<table>
  <tr>  <td></td>         <td>scikit-learn</td>  <td>GPU</td>   <td>Nearist</td>  </tr>
  <tr>  <td>K = 1</td>    <td>0.60</td>          <td>3.87</td>  <td>0.17</td>  </tr>
  <tr>  <td>K = 10</td>   <td>1.23</td>          <td>3.80</td>  <td>0.17</td>  </tr>
  <tr>  <td>K = 100</td>  <td>1.85</td>          <td>3.86</td>  <td>0.17</td>  </tr>
</table>  

Notice that the GPU is actually slower than the CPU when only presented with one query at a time. One reason is because GPUs are more efficient at performing matrix-matrix multiplication (multiple query vectors against the dataset) than they are at performing vector-matrix multiplication (a single query vector against the dataset).

Also notice how the GPU implementation is relatively unaffected by the value of _k_, whereas scikit-learn performance decreases with larger values of _k_. The scikit-learn behavior suggests that they might be optimizing the sorting technique based on the value of _k_.

The Nearist hardware also is relatively unaffected by the value of _k_; this is due to efficient parallelization of the sorting process in the Nearist architecture. Increasing _k_ does increase the Nearist latency, but the effect is only on the order of a few milliseconds, and so it isn’t visible here.

### Throughput Measurements
The next table shows the average throughput (in queries per second) achieved by each of the different platforms when presented with an input batch of 100 query vectors. 

The throughput tends to be higher than just 1 / latency. This is because the PC, GPU, and Nearist hardware all have the ability to process multiple queries in parallel to varying degrees. The throughput is reported as the average number of queries per second (calculated as 100 queries / total processing time).

The measurements are all averaged over 5 runs. 

_Throughput in queries / second_

<table>
  <tr>  <td></td>         <td>scikit-learn</td>   <td>GPU</td>    <td>Nearist</td>  </tr>
  <tr>  <td>K = 1</td>    <td>26.15</td>          <td>25.74</td>  <td>94.12</td>  </tr>
  <tr>  <td>K = 10</td>   <td>10.51</td>          <td>26.32</td>  <td>94.12</td>  </tr>
  <tr>  <td>K = 100</td>  <td>6.61</td>           <td>20.40</td>  <td>94.12</td>  </tr>
</table>  


The throughput numbers show the same behaviors surrounding the value of _k_, with the CPU performance being the most impacted by the value of _k_. 

This is also where we start to see the GPU outperforming the CPU, when it is able to operate on more query vectors simultaneously.

## Conclusions
The Nearist prototype board only has 1/10th of the performance of the production version, but already shows a more than 10x improvement over the CPU in throughput and latency for k-NN with k=100. Against the GPU, the Nearist prototype board has more than 20x better latency, and more than 4x greater throughput at k=100.

In the coming months, we will update this post with GPU measurements using both chips on the Tesla K80 card (which should give a roughly 2x improvement), and we’ll also update the Nearist results using the full-performance production card (which should give roughly a 10x improvement).

In our next benchmarking post, we will look at a classification task which will allow us to compare accuracy in addition to the throughput and latency. We will also add results using the Annoy library in Python, which uses approximate nearest neighbor techniques.
