---
layout: post
title:  "Concept Search on Wikipedia"
date:   2017-02-22 7:00:00 -0800
comments: true
image: assets/wikipedia/banner.png
tags: gensim, python, wikipedia, corpus, concept search, similarity search, nlp
---

I recently created a project on GitHub called [wiki-sim-search](https://github.com/chrisjmccormick/wiki-sim-search) where I used gensim to perform concept searches on English Wikipedia.

`gensim` includes a script, `make_wikicorpus.py`, which converts all of Wikipedia into vectors. They've also got a nice tutorial on using it [here](https://radimrehurek.com/gensim/wiki.html). 

I started from this gensim script and modified it heavily to comment and organize it, and achieve some more insight into each of the steps. I also included a few additional steps, like training and applying the LSI model, and performing similarity searches by article title.

## What it takes
Building a corpus from Wikipedia is a pretty lengthy process--about 12 hours from Wikipedia dump to a set of LSI vectors. It also uses a lot of hard disk space, as you can imagine.

Here's a breakdown of the steps involved in my version of `make_wikicorpus.py`, along with the files generated and their sizes.

These numbers are from running on my desktop PC, which has an Intel Core i7 4770, 16GB of RAM, and an SSD.

<table>
<tr><td>#</td><td>Step</td><td>Time (h:m)</td><td>Output File</td><td>File Size</td></tr>
<tr><td>0</td><td>Download Wikipedia Dump</td><td>--</td><td>enwiki-latest-pages-articles.xml.bz2</td><td>12.6 GB</td></tr>
<tr><td>1</td><td>Parse Wikipedia & Build Dictionary</td><td>3:12</td><td>dictionary.txt.bz2</td><td>769 KB</td></tr>
<tr><td>2</td><td>Convert articles to bag-of-words vectors</td><td>3:32</td><td>bow.mm</td><td>9.44 GB</td></tr>
<tr><td>2a.</td><td>Store article titles</td><td>--</td><td>bow.mm.metadata.cpickle</td><td>152 MB</td></tr>
<tr><td>3</td><td>Learn tf-idf model from document statistics</td><td>0:47</td><td>tfidf.tfidf_model</td><td>4.01 MB</td></tr>
<tr><td>4</td><td>Convert articles to tf-idf</td><td>1:40</td><td>corpus_tfidf.mm</td><td>17.9 GB</td></tr>
<tr><td>5</td><td>Learn LSI model with 300 topics</td><td>2:07</td><td>lsi.lsi_model</td><td>3.46 MB</td></tr>
<tr><td></td><td></td><td></td><td>lsi.lsi_model.projection</td><td>3 KB</td></tr>
<tr><td></td><td></td><td></td><td>lsi.lsi_model.projection.u.npy</td><td>228 MB</td></tr>
<tr><td>6</td><td>Convert articles to LSI</td><td>0:58</td><td>lsi_index.mm</td><td>1 KB</td></tr>
<tr><td></td><td></td><td></td><td>lsi_index.mm.index.npy</td><td>4.69 GB</td></tr>
<tr><td></td><td><strong>TOTALS</strong></td><td><strong>12:16</strong></td><td></td><td><strong>45 GB</strong></td></tr>
</table>

The original gensim script stops after step 4.

Notice how it parses Wikipedia twice--in steps 1 and steps 2. The alternative would be that, on the first pass, you write out the extracted tokens to another file (there's no way you could keep them all in memory). If you want to save a little bit of time, I included my compiled dictionary in the project, so that you can skip over step 1.

## Insights into Wikipedia

It's fun to look at the statistics on Wikipedia. 

These statistics are from the Wikipedia dump that I pulled down on 1/18/17.

<table>
<tr><td>17,180,273</td><td>Total number of articles (without any filtering)</td></tr>
<tr><td>4,198,780</td><td>Number of articles after filtering out "article redirects" and "short stubs"</td></tr>
<tr><td>2,355,066,808</td><td>Total number of tokens in all articles (without any filtering)</td></tr>
<tr><td>2,292,505,314</td><td>Total number of tokens after filtering articles</td></tr>
<tr><td>8,746,676</td><td>Total number of unique words found in all articles (*after* filtering articles)</td></tr>
</table>

In summary:

* There are ~4.2M Wikipedia articles with real content.
* There are ~2.3B words across all of these articles, which means the average article length is 762 words.
* There are 8.75M unique words in Wikipedia.

## Similarity Search
Once you have the LSI vectors, you can search wikipedia to find the most similar articles to a specified article.

As a fun example, I searched for the top 10 articles conceptually similar to [Topic model](https://en.wikipedia.org/wiki/Topic_model).

The results look pretty reasonable to me:

{% highlight text %}
Most similar documents:
  0.90    Online content analysis
  0.90    Semantic similarity
  0.89    Information retrieval
  0.89    Data-oriented parsing
  0.89    Concept search
  0.89    Object-role modeling
  0.89    Software analysis pattern
  0.88    Content analysis
  0.88    Adaptive hypermedia
  0.88    Model-driven architecture
{% endhighlight %}

It's interesting, I didn't know about most of these related articles. I had never heard of "Online content analysis", and wouldn't have thought to look at "Semantic similarity" in researching topic modeling. A concept search like this seems pretty helpful for researching topics.

## gensim Web App
The gensim guys turned this concept search of Wikipedia functionality into a cool little web app [here](https://rare-technologies.com/performance-shootout-of-nearest-neighbours-querying/#bonus_application).

It uses an approximate nearest neighbor library called Annoy to deliver fast results.

If you search by the article 'Topic model' using their web app, the results don't seem nearly as good:

{% highlight text %}
    Topic model
    Knowledge modeling
    Technology acceptance model
    Anchor Modeling
    Articulatory synthesis
    Technology adoption lifecycle
    MMT (Eclipse)
    Resource Description Framework
    Geotechnical centrifuge modeling
    Trans-Proteomic Pipeline
{% endhighlight %}

## Top words per topic
It's interesting to look at the top words per topic. You can see the top 10 words for each of the 300 learned topics [here](https://github.com/chrisjmccormick/wiki-sim-search/blob/master/topic_words.txt).

Some of them make sense:

* Topic #37: democratic, republican, trump, hillary, racing, airport, pt, huckabee, obama, clinton,
* Topic #51: ef, tornado, tropical, airport, church, damage, utc, storm, url, fc, 
* Topic #114: forests, stm, broadleaf, shrublands, estero, subtropical, palearctic, grasslands, moist, utc,

But I'd say most of the topics are pretty confusing. There are a lot of words that show up which don't seem like they should be nearly so important. For example:

* 'saleen' (a car manufacturer) appears in 15 topics
* 'astragalus' (a species of plant) appears in 7
* 'southampton' (a city in England) appears in 15 topics

There are also a lot of short words that seem like they may be HTML or stlying tags that are somehow not getting parsed out. For example, 'http', 'ft', 'bgcolor', 'alt'.

I'm hoping to dig into the parsing code a bit and see if I can't improve the results.

## Memory Concerns

Performing similarity searches on Wikipedia gets interesting because of the matrix size. 

I used LSI with 300 topics, so the matrix is:

{% highlight text %}
([4.2E6 articles] x [300 features/article] x [4 bytes/feature]) / [2^30 bytes/GB] = 4.69GB
{% endhighlight %}

That's a lot of RAM!

For my implementation, I used `gensim.similarities.MatrixSimilarity`, which loads the entire set of LSI vectors into memory.

However, gensim also includes a class `gensim.similarities.Similarity` that allows you to "shard" the matrix so that you can keep it on disk and process it in chunks.

It would be interesting to try this and compare search performance.
