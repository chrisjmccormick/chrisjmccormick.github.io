---
layout: post
title:  "Google's trained Word2Vec model in Python"
date:   2016-04-12 22:00:00 -0800
comments: true
categories: tutorials
tags: gensim, Python, Windows, Text Classification, Natural Language Processing, Latent Semantic Analysis, Latent Semantic Indexing, SVD, tf-idf
---

In this post I'm going to describe how to get Google's *pre-trained* Word2Vec model up and running in Python to play with. 

As an interface to word2vec, I decided to go with a Python package called gensim. gensim appears to be a popular NLP package, and has some nice documentation and tutorials, including for word2vec.

You can download Google's pre-trained model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing "Google's pre-trained Word2Vec model"). It's 1.5GB! It includes word vectors for a vocabulary of 3 million words and phrases. The vector length is 300 features. 

Loading this model using gensim is a piece of cake; you just need to pass in the path to the model file.

{% highlight py %}
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  
{% endhighlight %}

However, if you're running 32-bit Python (like I was) you're going to get a memory error! 

This is because gensim allocates a big matrix to hold all of the word vectors, and if you do the math...

3 million words * 300 features * 4bytes/feature = ~3.35GB

...that's a big matrix!

Assuming you've got a 64-bit machine and a decent amount of RAM (I've got 16GB; maybe you could get away with 8GB?), your best is to switch to 64-bit Python.

64-bit Python on Windows
========================
It took me some effort get a 64-bit Python setup with gensim up and running, so I thought I'd share my steps.

I had been using Python(x, y) to get a nice machine learning-oriented Python environment up and running. However, there doesn't appear to be a 64-bit release of Python(x, y).

I found a package called WinPython that does include 64-bit support. It looks to be actively supported, and includes all of the features I cared about from Python(x, y) (it includes the Spyder IDE and scikit-learn with all its dependencies).

Head over to the homepage for WinPython [here](https://winpython.github.io/ "WinPython homepage"). WinPython currently has both a Python 3.4 and 3.5 version; gensim requires 3.5 so make sure to grab that one. As of April 2016, the current latest version is "WinPython-64bit-3.5.1.2.exe"--here's a [direct link](https://sourceforge.net/projects/winpython/files/WinPython_3.5/3.5.1.2/WinPython-64bit-3.5.1.2.exe/download, "WinPython 64-bit Python 3.5") to that version.

You can extract WinPython wherever you want; I put mine right under C:\.

WinPython doesn't put itself in the Windows registry or on the system path; however, it does include some batch scripts for doing this. Look under `C:\WinPython-64bit-3.5.1.2\scripts\` and you'll find `register_python.bat` and `env.bat`

Open a Windows command prompt and run those two batch scripts. Then, in the same command window, you can install gensim easily by executing the following on the command line: `easy_install -U gensim`

###Issues with Theano
You may or may not run into this--I may have done something screwy to cause this problem in my setup. 

After installation, when I tried to import gensim, I got an error about not having g++ setup. This is actually coming from Theano, which gensim borrows a package from. You can go figure out how to get Theano set up properly, or you can just not use it. I opted for the latter and just commented out the lines in `shared_corpus.py` that attempt to pull in Theano.
