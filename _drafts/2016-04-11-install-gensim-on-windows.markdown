---
layout: post
title:  "Install gensim on Windows"
date:   2016-04-11 22:00:00 -0800
comments: true
categories: tutorials
tags: gensim, Python, Windows, Text Classification, Natural Language Processing, Latent Semantic Analysis, Latent Semantic Indexing, SVD, tf-idf
---

I wanted to run an experiment with word2vec, so I installed the 'gensim' Python library.

You can follow the installation instructions on the gensim homepage [here](https://radimrehurek.com/gensim/install.html "gensim installation instructions"). If you're installing on Windows, it's just awkward enough that I thought I'd share the specific steps I took.

*Step 1:* The gensim installation is done using a setup utility called 'setuptools' that you'll need to grab. Just download [this python script](https://bootstrap.pypa.io/ez_setup.py "setuptools installation script"). Doesn't matter where you save it, I just put it on my desktop. Then use Python to run the script. 

For me, this meant opening the file in the spyder IDE and hitting run.

![Python console installing setuptools][setuptools_install]

Oddly, it throws an exception at the end as it exits, but if you see the above messages it worked fine.

*Step 2:* Now you have setuptools installed, and it's callable from the Windows command line.

There's a utility called PowerShell that comes with Windows (I had never heard of this!). Launch it, then run the following command (It doesn't appear to matter where you run it from). 

`easy_install --upgrade gensim`

![Install gensim from PowerShell][gensim_install]

I had to close down the spyder IDE and relaunch it in order for it to pick up the gensim package. After that, you can just run `import gensim.models` on the Python command line to make sure it finds the package.

And that's it!

[setuptools_install]: {{ site.url }}/assets/Python_install_setuptools.png
[gensim_install]: {{ site.url }}/assets/Powershell_install_gensim.png

