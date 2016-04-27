---
author: chrisjmccormick
comments: true
date: 2016-02-12 22:14:39 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2016/02/12/jeff-dean-keynote-at-gtc-2015/
slug: jeff-dean-keynote-at-gtc-2015
title: Jeff Dean Keynote at GTC 2015
wordpress_id: 6082
---

Nvidia puts on its GPU Technology Conference (GTC) each year to highlight work being done on GPUs outside of graphics--including machine learning. 

Last year, Jeff Dean from Google gave one of the keynotes, and you can watch it [here](http://www.gputechconf.com/highlights/2015-replays). It’s an hour long, but it’s pretty easy to digest.

Jeff shares a lot about how deep learning is being leveraged by Google--here are some of the insights that stood out to me.

_What deep learning can and can’t do currently_

If you study machine learning, you know that deep learning has been successful at solving some remarkably difficult tasks. However, you also know that we’re far from having computers that are able to tackle every task that a human can.

I like how Jeff Dean explained this distinction. He said we have great models now for solving a number of specific, difficult tasks, but that we still don’t have the right models for many other complex tasks. 

Here are some specific complex tasks he said we now have good models for:



	
  * Looking at an image of something and labeling it. For example, given a picture of lion, labeling it “lion”. 

	
  * Recognizing the basic utterances or syllables within speech, which can in turn be used for speech recognition.

	
  * Predicting how relevant a document or web page is for a query.

	
  * Machine translation -- For example, taking text in English and translating it to another language. 

	
  * Generate human-like descriptions of images. 


_Deep Learning at use in Google_

One question that’s often on mind--how much are these advances in deep learning providing real market value? Do any of these techniques actually provide value to Google’s business?

Jeff said that they’ve launched “more than 50” products with deep learning in the last 2 years (this presentation was given in 2015). Some examples given: photo search, Android speech recognition, StreetView, Ads placement.

Another interesting tidbit--Jeff said they launched their deep net for speech recognition in 2012, and that it uses a smaller net on the phone, and a bigger one back at the datacenter. The smaller one is lower latency (since there’s no communication overhead), but not as accurate. It wasn’t clear whether the smaller one is just for ‘offline’ recognition, or if the two networks serve complementary roles.

_GPU usage at Google_

Jeff said that Google “regularly” works on models with “dozens of machines” each of which might have 8 GPU cards; so they have “100s of GPU cards” computing on a single copy of a model. I thought that was a nice insight into the scale of their GPU use. 

I’d be curious to understand whether they’re using the highest performing Tesla cards, or if they’ve decided it’s better to use a larger number of cheaper cards. On one hand, I imagine that there are cards which have a better dollar / performance ratio than the pricey Tesla cards. But on the other hand, I would also think that it would get expensive (in equipment, maintenance, power, etc.) to have a larger number of servers and to coordinate between them. 

_Matrix-matrix vs. matrix-vector multiplication_

Jeff acknowledges that GPUs are able to accelerate matrix-matrix multiplication operations more efficiently than matrix-vector. 

This has been my experience as well in measuring GPU performance on some tasks. However, it’s not obvious to me why this should be the case--there’s still plenty of opportunity for parallel computation in a matrix-vector operation, so I don’t think that’s it. It must have more to do with memory access patterns. At any rate, it’s good to hear Jeff’s confirmation of this behavior.
