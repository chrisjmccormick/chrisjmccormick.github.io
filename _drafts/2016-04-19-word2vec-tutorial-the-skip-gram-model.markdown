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

Word2Vec uses a trick you may have seen elsewhere in machine learning. We're going to train a neural network with a single hidden layer to perform a certain task, but then we're not actually going to use that neural network for the task we trained it on!

<div class="message">
Another place you may have seen this trick is in unsupervised feature learning, where you train an auto-encoder to compress an input vector in the hidden layer, and decompress it back to the original in the output layer. After training it, you strip off the output layer (the decompression step) and just use the hidden layer--it's a trick for learning good image features without having labeled training data.
</div>

Let's get specific. I'm going to use some example values in place of variables. Let's say we have a vocabularly of 10,000 unique words, and we are going to train a Word2Vec model with 300 features.

Take the following sentence

"Do you want ants? Because that's how you get ants."

We're going to look at the word "ants" and the words immediately around it.

<table>
<tr><td>-2</td><td>-1</td><td>0</td><td>1</td><td>2</td></tr>
<tr><td>you</td><td>want</td><td>ants</td><td>because</td><td>that's</td></tr>
</table>

Side note: Stop words may or may not be removed when training Word2Vec. The pre-trained model released by Google (3 million word vectors learned from 100 billion words of Google news) does not include stop words. On the other hand, this [tutorial at Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors "Kaggle tutorial on Word2Vec") says it is good to leave them in.

We're going to train the neural network to predict, for a given input word, the probabilities of different words appearing nearby it ("Nearby" is formally defined by a "window size" parameter that dictates how many words behind and ahead of the input word we're looking at).  

So how is this all represented?

We're going to represent the input word "ants" as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) and we'll place a "1" in the position corresponding to the word "ant", and 0s in all of the other positions.

The output of the network is a single vector containing, for every word in our vocabulary, the probability that each word would appear near the input word. 

Here's the architecture of our neural network.

[![Skip-gram Neural Network Architecture][skip_gram_net_arch]][skip_gram_net_arch]

There is no activation function on the hidden layer neurons, but the output neurons use softmax. We'll come back to this later.

The Hidden Layer
================

The hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

If you look at the *rows* of this weight matrix, these are actually what will be our word vectors!

[![Hidden Layer Weight Matrix][weight_matrix]][weight_matrix]

So the end goal of all of this is really just to learn this hidden layer weight matrix -- the output layer we'll just toss when we're done!

Let's get back, though, to working through the definition of this model that we're going to train.

Now, you might be asking yourself--"That one-hot vector is almost all zeros... what's the effect of that?" If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, it will effectively just *select* the matrix row corresponding to the "1". Here's a small example to give you a visual.

[![Effect of matrix multiplication with a one-hot vector][matrix_mult_w_one_hot]][matrix_mult_w_one_hot]

This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the "word vector" for the input word.

The Output Layer
================

The word vector then gets fed to the output neurons. Remember that for our simple example we just have four output neurons, one for each of the four surrounding word positions.

<div class="message">
The number of surrounding words you take into account is referred to as the window size. Our window size of 2 (2 words to either direction --> 4 output nodes) is tiny. A more typical value would be 10 (20 output neurons).
</div>

Let's focus on just the output neuron for the word at position -2.

The input to ouput neuron -2 is simply the `1 x 300` word vector for "ants". The weight matrix for output neuron -2 is `300 x 10,000` -- just like the hidden layer, except now there is one column per word instead of one row per word. 

Now let's say the word car is at position 500 in our vocabulary. When we multiply our word vector for "ants" with the column for the word "car", we are calculating a value that reflects how likely it is for "car" to appear two positions to the left of "ants".

[![Behavior of the output neuron for position -2][output_neuron]][output_neuron]

Intuition
=========
Ok, are you ready for an exciting bit of insight into this network? 

If two different words have very similar "contexts" (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. And one way for the network to output similar context predictions for these two words is if *the word vectors are similar*. 

Ta da! 

If two words have similar contexts, then our network is motivated to learn similar word vectors for these two words.

And what does it mean for two words to have similar contexts? Probably that they are synonyms or just closely related in some way.

[skip_gram_net_arch]: {{ site.url }}/assets/word2vec/skip_gram_net_arch.png
[weight_matrix]: {{ site.url }}/assets/word2vec/word2vec_weight_matrix_lookup_table.png
[matrix_mult_w_one_hot]: {{ site.url }}/assets/word2vec/matrix_mult_w_one_hot.png
[output_neuron]: {{ site.url }}/assets/word2vec/output_weights_function.png
