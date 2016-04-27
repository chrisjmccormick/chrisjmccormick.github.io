---
layout: post
title:  "Word2Vec Tutorial - The Skip-Gram Model"
date:   2016-04-19 22:00:00 -0800
comments: true
categories: tutorials
tags: Word2Vec, Skip Gram, tutorial, neural network, NLP, word vectors
---

This tutorial covers the skip gram neural network architecture for Word2Vec. My intention with this tutorial was to skip over the usual introductory and abstract insights about Word2Vec, and get into more of the details. Specifically here I'm diving into the skip gram neural network model.

The Model
=========
The skip-gram neural network model is actually surprisingly simple in its most basic form; I think it's the all the little tweaks and enhancements that start to clutter the explanation.

Let's start with a high-level insight about where we're going. Word2Vec uses a trick you may have seen elsewhere in machine learning. We're going to train a simple neural network with a single hidden layer to perform a certain task, but then we're not actually going to use that neural network for the task we trained it on! Instead, the goal is actually just to learn the weights of the hidden layer--we'll see that these weights are actually the "word vectors" that we're trying to learn.

<div class="message">
Another place you may have seen this trick is in unsupervised feature learning, where you train an auto-encoder to compress an input vector in the hidden layer, and decompress it back to the original in the output layer. After training it, you strip off the output layer (the decompression step) and just use the hidden layer--it's a trick for learning good image features without having labeled training data.
</div>

The Fake Task
=============
So now need to talk about this "fake" task that we're going to build the neural network to perform, and then we'll come back later to how this indirectly gives us those word vectors that we are really after.

We're going to train the neural network to tell us, for a given word in a sentence, what is the probability of each and every other word in our vocabulary appearing anywhere within a small window around the input word. For example, if you gave the trained network the word "Soviet" it's going to say that words like "Union" and "Russia" have a high probability of appearing nearby, and unrelated words like "watermelon" and "kangaroo" have a low probability. And it's going to output probabilities for every word in our vocabulary!

<div class="message">
When I say "nearby", we'll actually be using a fixed "window size" that's a parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead (10 in total).
</div>

We're going to train the neural network to do this by feeding it word pairs found in our training documents. The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of ("Soviet", "Union") than it is of ("Soviet", "Sasquatch"). When the training is finished, if you give it the word "Soviet" as input, then it will output a much higher probability for "Union" or "Russia" than it will for "Sasquatch".

Model Details
=============

So how is this all represented?

First of all, you know you can't feed a word just as a text string to a neural network, so we need a way to represent the words to the network. To do this, we first build a vocabulary of words from our training documents--let's say we have a vocabulary of 10,000 unique words.

We're going to represent an input word like "ants" as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) and we'll place a "1" in the position corresponding to the word "ants", and 0s in all of the other positions.

The output of the network is a single vector containing, for every word in our vocabulary, the probability that each word would appear near the input word. 

Here's the architecture of our neural network.

[![Skip-gram Neural Network Architecture][skip_gram_net_arch]][skip_gram_net_arch]

There is no activation function on the hidden layer neurons, but the output neurons use softmax. We'll come back to this later.

The Hidden Layer
================

For our example, we're going to say that we're learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

If you look at the *rows* of this weight matrix, these are actually what will be our word vectors!

[![Hidden Layer Weight Matrix][weight_matrix]][weight_matrix]

So the end goal of all of this is really just to learn this hidden layer weight matrix -- the output layer we'll just toss when we're done!

Let's get back, though, to working through the definition of this model that we're going to train.

Now, you might be asking yourself--"That one-hot vector is almost all zeros... what's the effect of that?" If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, it will effectively just *select* the matrix row corresponding to the "1". Here's a small example to give you a visual.

[![Effect of matrix multiplication with a one-hot vector][matrix_mult_w_one_hot]][matrix_mult_w_one_hot]

This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the "word vector" for the input word.

The Output Layer
================

The `1 x 300` word vector for "ants" then gets fed to the output layer. The output layer is a softmax regression classifier. There's an in-depth tutorial on Softmax Regression [here](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/), but the gist of it is that each output neuron (one per word in our vocabulary!) will produce an output between 0 and 1, and the sum of all these output values will add up to 1. 

Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, then it applies the function `exp(x)` to the result. Finally, in order to get the outputs to sum up to 1, we divide this result by the sum of the results from *all* 10,000 output nodes.

Here's an illustration of calculating the output of the output neuron for the word "car".

[![Behavior of the output neuron][output_neuron]][output_neuron]

Intuition
=========
Ok, are you ready for an exciting bit of insight into this network? 

If two different words have very similar "contexts" (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. And one way for the network to output similar context predictions for these two words is if *the word vectors are similar*. So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms like "intelligent" and "smart" would have very similar contexts. Or that words that are related, like "engine" and "transmission", would probably have similar contexts as well. 

This can also handle stemming for you -- the network will likely learn similar word vectors for the words "ant" and "ants" because these should have similar contexts.

[skip_gram_net_arch]: {{ site.url }}/assets/word2vec/skip_gram_net_arch.png
[weight_matrix]: {{ site.url }}/assets/word2vec/word2vec_weight_matrix_lookup_table.png
[matrix_mult_w_one_hot]: {{ site.url }}/assets/word2vec/matrix_mult_w_one_hot.png
[output_neuron]: {{ site.url }}/assets/word2vec/output_weights_function.png
