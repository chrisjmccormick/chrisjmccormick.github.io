---
layout: post
title:  "Word2Vec Tutorial Part 2"
date:   2017-01-20 22:00:00 -0800
comments: true
image: /assets/word2vec/original_paper.png
tags: Word2Vec, NLP, word vectors
---


In part 2 of the word2vec tutorial, I’ll cover a few additional modifications to the basic skip-gram model which are important for actually making it feasible to train.

When you read the tutorial on the skip-gram model for Word2Vec, you may have noticed something--it’s a huge neural network! 

In the example I gave, we had word vectors with 300 components, and a vocabulary of 10,000 words. Recall that the neural network had two weight matrices--a hidden layer and output layer. Both of these layers would have a weight matrix with 300 x 10,000 = 3 million weights each!

Running gradient descent on a neural network that large is going to be slow. And to make matters worse, you need a huge amount of training data in order to tune that many weights and avoid over-fitting. millions of weights times billions of training samples means that training this model is going to be a beast.

The authors of Word2Vec addressed these issues in their second [paper](http://arxiv.org/pdf/1310.4546.pdf).

There are three innovations in this second paper:

1. Treating common word pairs or phrases as single "words" in their model.
2. Subsampling frequent words to decrease the number of training examples.
3. Modifying the optimization objective with a technique they called "Negative Sampling", which causes each training sample to update only a small percentage of the model's weights.

It's worth noting that subsampling frequent words and applying Negative Sampling not only reduced the compute burden of the training process, but also improved the quality of their resulting word vectors as well.

Word Pairs and "Phrases"
========================
The authors pointed out that a word pair like "Boston Globe" (a newspaper) has a much different meaning than the individual words "Boston" and "Globe". So it makes sense to treat "Boston Globe", wherever it occurs in the text, as a single word with its own word vector representation.

Their paper doesn't go in to detail about how they performed phrase detection, other than to say that it was a "data driven" approach.

You can see the results, though, in their published model, which was trained on 100 billion words from a Google News dataset. The addition of phrases to the model swelled the vocabulary size to 3 million words! 

If you're interested in their resulting vocabulary, I poked around it a bit and published a post on it [here](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/). You can also just browse their vocabulary [here](https://github.com/chrisjmccormick/inspect_word2vec/tree/master/vocabulary).

Subsampling Frequent Words
==========================



