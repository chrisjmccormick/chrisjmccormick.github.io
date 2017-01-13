---
layout: post
title:  "Word2Vec Resources"
date:   2016-04-27 22:00:00 -0800
comments: true
image: /assets/word2vec/original_paper.png
tags: Word2Vec, NLP, word vectors
---

While researching Word2Vec, I came across a lot of different resources of varying usefullness, so I thought I'd share my collection of links and notes on what they contain.

* TOC
{:toc}

Original Papers & Resources from Google Team
============================================

Word2Vec was presented in two initial papers released within a month of each other. The original authors are a team of researchers from Google.

### Efficient Estimation of Word Representations in Vector Space
[Link to paper](http://arxiv.org/pdf/1301.3781.pdf)

This was the first paper, dated September 7th, 2013.

This paper introduces the Continuous Bag of Words (CBOW) and Skip-Gram models. However, don't expect a particularly thorough description of these models in this paper...

I believe the reason for this is that these two new models are presented more as modifications to previously existing models for learning word vectors. Some of the terminology and concepts in this Word2Vec paper come from these past papers and are not redifined in Google's paper.

A good example are the labels "projection layer" and "hidden layer" which come from the "NNLM" model. The term "projection layer" is used to refer to a middle layer of the neural network *with no activation function*, whereas "hidden layer" implies a non-linear activation. 

### Distributed Representations of Words and Phrases and their Compositionality
[Link to paper](http://arxiv.org/pdf/1310.4546.pdf)

This was a follow-up paper, dated October 16th, 2013.

This paper adds a few more innovations which address the high compute cost of training the skip-gram model on a large dataset. These added tweaks are fundamental to the word2vec algorithm, and are implemented in Google's C version as well as the Python implementation in `gensim`.

These innovations are:
1. Subsampling common words (that is, eliminating some training samples).
2. "Negative Sampling" - A modification of the optimization objective which causes each training sample to update only a small percentage of the model's weights.

Additionally, they point out the value in recognizing common "phrases" and treating them as single words in the model (e.g., "United_States" or "New_York").

###  Presentation on Word2Vec
[Link to presentation](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit)

This was presented December 9th, 2013 at NIPS 2013 by Tomas Mikolov from Google.

I think this is mainly a re-hash of the content in the two papers. Seeing it presented differently may help you pull out some additional insights, though.

### C Code Implementation
[Link to code](https://code.google.com/archive/p/word2vec/)

The above link is to the home page for google's own Word2Vec implementation in C.

You can also find here some pre-trained models that they have provided. Note that it's possible to load these pre-trained models into `gensim` if you want to work with them in Python.

Tutorials
=========

### Alex Minnaar's Tutorials
The best tutorials I found online were done by [Alex Minnaar](http://alexminnaar.com/). 

He's since taken the tutorials down, but I have PDF copies here:

* [Part I - The Skip-Gram Model]({{ site.url }}/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_I_The_Skip-Gram_Model.pdf)
* [Part II - Continuous Bag-of-Words Model]({{ site.url }}/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)

### Kaggle Word2Vec Tutorial
[Link to tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

This is pretty cool. It's a Kaggle competition that's really just a Python tutorial to teach you about using Word2Vec with `gensim`. It's well written and will walk you through all of the steps carefully. It does very little to explain the algorithms used, but is great on the practical implementation side.

It uses a sentiment analysis task (on the IMDB movie review dataset) as an example project. While the tutorial is great for showing you how to get set up with `gensim` and even train your own Word2Vec model on the data, you'll discover that it essentially fails at applying Word2Vec effectively on the example task of sentiment analysis! To get good results on the IMDB dataset, you'll want to check out Google's Doc2Vec technique (which isn't covered in this tutorial).  

Here's what the tutorial covers. 
 
Part 1:

* Cleaning and tokening the text data.
* Vectorizing the documents using word counts.
* Classification using a random forest.

Part 2:

* Setting up `gensim`
* Training a Word2Vec model (learning word vectors from the dataset) using `gensim`

Part 3:

* This section attempts two rather unsuccessful ways of applying the word vectors to create vector representations of each review. Neither manages to outperform the simpler word-count approach from part 1.
    * Creating vector representations of each movie review by averaging its word vectors.
    * Clustering the word vectors to identify sets of synonyms, then using the word-count approach, but this time combining synonyms into a single bucket. 

Part 4:

* Points to Google's Doc2Vec as a superior solution to this task, but doesn't provide implementation details.

### Folgert Karsdorp's Word2Vec Tutorial 
[Link to tutorial](http://www.folgertkarsdorp.nl/word2vec-an-introduction/)

I haven't read this tutorial in depth... It covers the Continuous Bag of Words model (instead of the Skip-Gram model). It even includes some of the backprop equations.

### Discussions on Quora

* [https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures-in-laymans-terms](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures-in-laymans-terms)
* [https://www.quora.com/How-does-word2vec-work](https://www.quora.com/How-does-word2vec-work)
* [https://www.quora.com/What-are-some-interesting-Word2Vec-results/answer/Omer-Levy](https://www.quora.com/What-are-some-interesting-Word2Vec-results/answer/Omer-Levy)

Implementations
===============
The below implementations also include some tutorials; I haven't gone through them in detail yet.

* Word2Vec and Doc2Vec in Python in gensim [here](http://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim/) and [here](http://rare-technologies.com/word2vec-tutorial/)
* Word2vec in Java in [deeplearning4j](http://deeplearning4j.org/word2vec.html)
* Java version from [Medallia](https://github.com/medallia/Word2VecJava)
* Word2vec implementation in [Spark MLlib](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec)
* Word2Vec implementation / tutorial in Google's [TensorFlow](https://www.tensorflow.org/versions/r0.8/tutorials/word2vec/index.html)

My Own Stuff
============

* I have my own tutorial on the skip-gram model of Word2Vec [here](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/).
* [Part 2](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) of my tutorial covers subsampling of frequent words and the Negative Sampling technique.
* I created a project called [inspec_word2vec](https://github.com/chrisjmccormick/inspect_word2vec) that uses gensim in Python to load up Google's large pre-trained model, and inspect some of the details of the vocabulary.
* I'm working on a Matlab implementation of Word2Vec, [word2vec_matlab](https://github.com/chrisjmccormick/word2vec_matlab). My goal is less about practical useage and more about understanding the model. For now, it doesn't support the most important part--actually training a Word2Vec model. What it does do currently is allow you to play with a paired-down (or, really, cleaned-up!) version of Google's pre-trained model in Matlab.