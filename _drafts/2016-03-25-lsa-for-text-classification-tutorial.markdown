---
layout: post
title:  "Latent Semantic Analysis (LSA) for Text Classification Tutorial"
date:   2016-03-25 22:00:00 -0800
comments: true
categories: tutorials
tags: Text Classification, Natural Language Processing, Latent Semantic Analysis, Latent Semantic Indexing, SVD, tf-idf
---

In this post I'll provide a tutorial of Latent Semantic Indexing as well as some Python example code that shows the technique in action.

Why LSA?
--------
Latent Semantic Analysis is a technique for creating a vector representation of a document. Having a vector representation of a document gives you a way to compare documents for their similarity by calculating the distance between the vectors. This in turn means you can do handy things like classifying documents to determine which of a set of known topics they most likely belong to.

Classification implies you have some known topics that you want to group documents into, and that you have some labelled training data. If you want to identify natural groupings of the documents without any labelled data, you can use clustering (see my post on clustering with LSA here).

tf-idf
------
The first step in LSA is actually a separate algorithm that you may already be familiar with.

LSA
---

https://en.wikipedia.org/wiki/Latent_semantic_analysis

 
Code
----

http://scikit-learn.org/stable/modules/neighbors.html#classification

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

