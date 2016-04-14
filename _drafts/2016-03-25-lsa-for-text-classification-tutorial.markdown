---
layout: post
title:  "Latent Semantic Analysis (LSA) for Text Classification Tutorial"
date:   2016-03-25 22:00:00 -0800
comments: true
categories: tutorials
tags: Text Classification, Natural Language Processing, Latent Semantic Analysis, Latent Semantic Indexing, SVD, tf-idf
---

In this post I'll provide a tutorial of Latent Semantic Analysis as well as some Python example code that shows the technique in action.

Why LSA?
--------
Latent Semantic Analysis is a technique for creating a vector representation of a document. Having a vector representation of a document gives you a way to compare documents for their similarity by calculating the distance between the vectors. This in turn means you can do handy things like classifying documents to determine which of a set of known topics they most likely belong to.

Classification implies you have some known topics that you want to group documents into, and that you have some labelled training data. If you want to identify natural groupings of the documents without any labelled data, you can use clustering (see my post on clustering with LSA [here](https://chrisjmccormick.wordpress.com/2015/08/05/document-clustering-example-in-scikit-learn/).

tf-idf
------
The first step in LSA is actually a separate algorithm that you may already be familiar with. It's called [term frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf "tf-idf on Wikipedia"), or tf-idf for short. 

tf-idf is pretty simple and I won't go into it here, but the gist of it is that each position in the vector corresponds to a different word, and you represent a document by counting the number of times each word appears. Additionally, you normalize each of the word counts by the frequency of that word in your overall document collection, to give less frequent terms more weight.

There's some thorough material on tf-idf in the Stanford NLP course on Coursera [here](https://class.coursera.org/nlp/lecture "Stanford NLP course on Coursera")--specifically, check out the lectures under "Week 7 - Ranked Information Retrieval". Or if you prefer some (dense) reading, you can check out the tf-idf chapter of the Stanford NLP textbook [here](http://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html "Stanford NLP textbook").

LSA
---
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis "LSA on Wikipedia") takes tf-idf one step further. Quite simply, you use SVD to perform dimensionality reduction on the tf-idf vectors. 

You might think to do this even if you had never heard of "LSA"--the tf-idf vectors tend to be long and unwieldy since they have one component for every word in the vocabulary. For instance, in my example Python code, these vectors have 10,000 components. So dimensionality reduction makes them more manageable for further operations like clustering or classification.

However, the SVD step does more than just reduce the computational load--you are trading a large number of features for a smaller set of *better* features. 

What makes the LSA features better?





 
Code
----

http://scikit-learn.org/stable/modules/neighbors.html#classification

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
