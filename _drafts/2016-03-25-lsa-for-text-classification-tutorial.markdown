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
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis "LSA on Wikipedia") takes tf-idf one step further. 

Side note: "Latent Semantic Analysis (LSA)" and "Latent Semantic Indexing (LSI)" are the same thing, with the latter name being used sometimes when referring specifically to indexing a collection of documents for search ("Information Retrieval").

LSA is quite simple, you just use SVD to perform dimensionality reduction on the tf-idf vectors, that's all there is to it.

You might think to do this even if you had never heard of "LSA"--the tf-idf vectors tend to be long and unwieldy since they have one component for every word in the vocabulary. For instance, in my example Python code, these vectors have 10,000 components. So dimensionality reduction makes them more manageable for further operations like clustering or classification.

However, the SVD step does more than just reduce the computational load--you are trading a large number of features for a smaller set of *better* features. 

What makes the LSA features better? 

### TODO

LSA Python Code
---------------
{% highlight text %}
Note: If you're less interested in learning LSA and just want to use it, you might consider checking out the nice `gensim` package in Python, it's built specifically for working with topic-modeling techniques like LSA.
{% endhighlight %}

I implemented an example of document classification with LSA in Python using scikit-learn. My code is available on GitHub, you can either visit the project page [here](https://github.com/chrisjmccormick/LSA_Classification "LSA_Classification project page"), or download the source [directly](https://github.com/chrisjmccormick/LSA_Classification/archive/master.zip "LSA_Classification direct download").

scikit-learn already includes a [document classification example](http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html "scikit-learn document classification example"). However, that example uses plain tf-idf rather than LSA, and is geared towards demonstrating batch training on large datasets. Still, I borrowed code from that example for things like retrieving the Reuters dataset.

I wanted to put the emphasis on the feature extraction and not the classifier, so I used simple [k-nn classification](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) with k = 5 (majority wins). 

Inspecting LSA
--------------
A little background on this Reuters dataset. These are news articles that were sent over the Reuters newswire in 1987. The dataset contains about 100 categories such as ‘mergers and acquisitions’, ‘interset rates’, ‘wheat’, ‘silver’ etc. Articles can be assigned multiple categories. The distribution of articles among categories is highly non-uniform; for example, the ‘earnings’ category contains 2,709 articles. And 75 of the categories contain less than 10 docs each!

Armed with that background, let's see what LSA is learning from the dataset.

You can look at component 0 of the SVD matrix, and look at the terms which are given the highest weight by this component. 

![Top 10 Terms in Component 0][top_10_terms]

These terms are all very common to articles in the "earnings" category.

Here's an example earnings article to give you some context:

{% highlight text %}
COBANCO INC CBCO> YEAR NET

Shr 34 cts vs 1.19 dlrs Net 807,000 vs 2,858,000 Assets 510.2 mln vs 479.7 mln Deposits 472.3 mln vs 440.3 mln Loans 299.2 mln vs 327.2 mln Note: 4th qtr not available. Year includes 1985 extraordinary gain from tax carry forward of 132,000 dlrs, or five cts per shr. Reuter
{% endhighlight %}

[top_10_terms]: {{ site.url }}/assets/Reuters_LSA_comp0_top10terms.png



