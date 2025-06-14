---
layout: post
title:  "The Inner Workings of Multihead Latent Attention (MLA)"
date:   2025-04-26 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1TkaHaLIG31pjUKYizLssDhnT_V364lJt
tags: Transformers, DeepSeek, DeepSeek V3, MLA, Attention, GPU
---

Multihead Latent Attention (MLA), introduced by DeepSeek in their V2 model, is an alternative to standard attention (and other variants such as MQA and GQA) which dramatically reduces memory bandwidth requirements for the attention calculations.
