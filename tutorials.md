---
layout: page
title: Tutorials
---

<hr/>

### BERT
[![BERT Architecture][bert_layers]][bert_layers]

* [BERT Research Series](https://www.youtube.com/playlist?list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6) on YouTube - Follow along in this 8 episode series as I make sense of how BERT works.
    * The culmination of this series was my [BERT eBook](https://bit.ly/2XNj5Ks)!
* BERT Application Examples:
    * Word Embeddings ([post][bert_word_embeddings], [notebook](https://colab.research.google.com/drive/1yFphU6PW9Uo6lmDly_ud9a6c4RCYlwdX))
    * Sentence Classification ([post][bert_sentence_classification], [notebook](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX))
    * Document Classification ([video](https://youtu.be/_eSGWNqKeeY), [notebook](https://bit.ly/2FcIdEb)
    * Named Entity Recognition ([notebook](https://bit.ly/3fKhvzo))
    * Multilingual BERT ([notebook](https://bit.ly/3itodLE))
    

<hr/>

[bert_layers]: {{ site.url }}/assets/BERT/bert_architecture_tutorials_page.png
[bert_word_embeddings]: {{ site.url }}/2019/05/14/BERT-word-embeddings-tutorial/
[bert_sentence_classification]: {{ site.url }}/2019/07/22/BERT-fine-tuning/

### Word2Vec

[![Skip-gram model][skip-gram_model]][skip-gram_model]

* [Word2Vec Tutorial - The Skip-Gram Model][word2vec_skip-gram]
* [Word2Vec Tutorial Part 2 - Negative Sampling][word2vec_negative_sampling]
* [Applying word2vec to Recommenders and Advertising][word2vec_recommenders]
* [Commented word2vec C code](https://github.com/chrisjmccormick/word2vec_commented)
* [Wor2Vec Resources][word2vec_res]


[skip-gram_model]: {{ site.url }}/assets/word2vec/skip_gram_net_arch.png
[word2vec_skip-gram]: {{ site.url }}/2016/04/19/word2vec-tutorial-the-skip-gram-model/
[word2vec_negative_sampling]: {{ site.url }}/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
[word2vec_recommenders]: {{ site.url }}/2018/06/15/applying-word2vec-to-recommenders-and-advertising/
[word2vec_res]: {{ site.url }}/2016/04/27/word2vec-resources/

<hr/>

### Radial Basis Function Networks

[![RBFN Architecture][rbfn_arch_small]][rbfn_arch_small]

I've written a number of posts related to Radial Basis Function Networks. Together, they can be taken as a multi-part tutorial to RBFNs.

* Part 1 - [RBFN Basics, RBFNs for Classification][rbfn_classification]
* Part 2 - [RBFN Example Code in Matlab][rbfn_code]
* Part 3 - [RBFN for function approximation][rbfn_func_approx]
* Advanced Topics:
  * [Gaussian Kernel Regression][kernel_regression]
  * [Mahalonobis Distance][mahal_dist]

[rbfn_arch_small]: {{ site.url }}/assets/rbfn/architecture_simple_small.png
[rbfn_classification]: {{ site.url }}/2013/08/15/radial-basis-function-network-rbfn-tutorial/ "RBFN basics and classification"
[rbfn_code]: {{ site.url }}/2013/08/16/rbf-network-matlab-code/ "Post on Matlab code for RBFN"
[rbfn_func_approx]: {{ site.url }}/2015/08/26/rbfn-tutorial-part-ii-function-approximation/ "RBFNs for function approximation"
[kernel_regression]: {{ site.url }}/2014/02/26/kernel-regression/ "Gaussian Kernel Regression"
[mahal_dist]: {{ site.url }}/2014/07/22/mahalanobis-distance/ "Mahalanobis Distance"

<hr/>

### Histograms of Oriented Gradients (HOG)

My tutorial and Matlab code on the HOG descriptor are easily one of the most popular items on my site.

* [HOG Person Detector Tutorial][hog_tutorial]
* [HOG Descriptor Matlab Code][hog_matlab]
* [HOG Result Clustering][hog_clustering]


[hog_tutorial]: {{ site.url }}/2013/05/09/hog-person-detector-tutorial/ "HOG Person Detector Tutorial"
[hog_matlab]: {{ site.url }}/2013/05/09/hog-descriptor-in-matlab/ "HOG descriptor in Matlab"
[hog_clustering]: {{ site.url }}/2013/11/07/opencv-hog-detector-result-clustering/ "HOG Detector result clustering"

<hr/>