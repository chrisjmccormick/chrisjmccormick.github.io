---
layout: post
title:  "The Inner Workings of Multihead Latent Attention (MLA)"
date:   2025-04-26 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1TkaHaLIG31pjUKYizLssDhnT_V364lJt
tags: Transformers, DeepSeek, DeepSeek V3, MLA, Attention, GPU
---

Multihead Latent Attention (MLA), introduced by DeepSeek in their V2 model, is an alternative to standard attention (and other variants such as MQA and GQA) which dramatically reduces memory bandwidth requirements for the attention calculations.

**Overview**

"Reducing bandwidth" means cutting down the number of memory reads required to perform the overall attention calculation.

Standard multihead attention creates distinct query, key, and value vectors for every head, each of which must be read into memory to calculate the attention scores. 

We'll see that MLA cleverly re-works the standard attention algebra and shifts all of the "per-head" calculations to the input side of the attention equations, allowing us to store a single 512-dimensional "latent" vector per context token. This single latent gets read into GPU memory and then **re-used** across all of the heads.

Remarkably, calculating attention using these 512-dimensional latents actually requires **4× more operations** than standard attention, but ultimately achieves a higher token generation throughput thanks to the dramatic reduction in memory reads.

The MLA formulation also requires us to change how we encode position information with RoPE. MLA introduces a concept of a "decoupled RoPE" embedding, but this alone doesn't resolve the problem. Rather, MLA reveals that position information can be sufficiently incorporated using just a single key head mapped to all query heads.

This post expands on these concepts, and takes a detailed look at the algebra, in particular.

**Notation**

_Dimensions_

Throughout this post I'll be using the actual dimensions of DeepSeek-V3, since I often find it easier to refer to / think about matrix shapes by their specific values instead of a sea of variables $d^h$, $d^c$, etc. I'll be using the base-2 notation of "K". In particular, DeepSeek-V3 has an embedding size of 7,168, so I'll write that as simply "7K", and a query compression dimension of 1,536, or "1.5K".

## Contents

* TOC
{:toc}
