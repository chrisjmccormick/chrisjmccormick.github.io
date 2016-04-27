---
author: chrisjmccormick
comments: true
date: 2015-05-08 23:38:36 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2015/05/08/experiences-renting-gpu-instances/
slug: experiences-renting-gpu-instances
title: Experiences Renting GPU Instances
wordpress_id: 6025
tags:
- Amazon
- GPU
- Machine Learning
- NVIDIA
---

I thought I'd share briefly some of our team's recent experiences in renting time on GPUs for machine learning work.

You've probably seen that GPUs are gaining popularity for machine learning because of their inherent parallelism. You may have also heard of renting GPU instances from Amazon for machine learning work. You might suppose, therefore, (as we did!) that renting an Amazon GPU instance would be a good way to gain access to a high-performance GPU.

Here is the key insight I wanted to share:


<blockquote>_The motivation for renting a GPU instance at Amazon* is not about getting access to a high performance GPU, but rather for the ability to cheaply gain access to many GPU instances for running parallel experiments._</blockquote>


*This applies to Amazon specifically--I'll discuss shortly our experience with other providers.

We discovered this insight empirically first. I implemented a benchmark test where we are calculating the L1 distance between a large number of vectors.

On my PC, I have a [GeForce GTX 660](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-660/specifications) ($190 in September, 2013), which has 960 CUDA cores and runs at 980MHz.

Amazon's GPU instance, named "g2.2xlarge", has a [GRID K520](http://www.nvidia.com/object/cloud-gaming-gpu-boards.html). It's a board from NVIDIA designed for "cloud gaming"--that is, multiple concurrent users. It has 2 GK104 GPUs with 1,536 CUDA cores each running at 800MHz.

Seems like it should deliver pretty decent performance, but the benchmark results showed otherwise.

To transfer the data to the card for the benchmark, it took 70ms on my PC and 290ms on the Amazon instance (~4.1x slower). For the actual calculations, it took about 190ms on my PC and 430ms on the Amazon instance (~2.3x slower).

What's the reason for this disparity? I don't know the detailed answer, but it has to do with the virtualization Amazon uses on their instances. [Netflix mentioned this issue](http://techblog.netflix.com/2014/02/distributed-neural-networks-with-gpus.html) in one of their blog posts:


<blockquote>“In a virtualized environment such as the AWS cloud, these accesses cause a trap in the hypervisor that results in even slower access.”</blockquote>


I also posted our issue to Reddit's Machine Learning community [here](http://www.reddit.com/r/MachineLearning/comments/305me5/slow_gpu_performance_on_amazon_g22xlarge/), and got some helpful replies. Essentially: (1) This is in fact what you should expect, and (2) The point of Amazon instances is to run multiple experiments in parallel at low cost.

Our goal was to benchmark our hardware against high-end GPUs, so we wanted full-speed access to something more potent than the GRID board at Amazon. Luckily, there are other services that will give you this. We landed on [Nimbix](http://www.nimbix.net/). Nimbix gives us "bare-metal" access (read, "no virtualization") to a Tesla K40. On the Nimbix machine, the calculation step of the benchmark completes about 4.8x faster than on my PC. Awesome!

The trade-off is it's more expensive - $5/hr. But for our purposes, it's great--we're running fairly short tests, and they charge you in fractions of an hour (that is, for the precise amount of time that you have the instance up).

Overall, the ability to rent GPU instances cheaply from Amazon for research is awesome. Just make sure you have the right performance expectations!
