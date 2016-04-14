---
layout: post
title:  "Document Classification with Word2Vec"
date:   2016-04-11 22:00:00 -0800
comments: true
categories: tutorials
tags: gensim, Python, Windows, Text Classification, Natural Language Processing
---

You can download Google's pre-trained model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing "Google's pre-trained Word2Vec model"). It's 1.5GB, so I can't include it in my GitHub project. Save it to the `model` directory of the project.

{highlight python}
# Load Google's pre-trained Word2Vec model.
model = Word2Vec.load_word2vec_format('/model/GoogleNews-vectors-negative300.bin', binary=True)
{endhighlight}


![Python console installing setuptools][setuptools_install]


[setuptools_install]: {{ site.url }}/assets/Python_install_setuptools.png



