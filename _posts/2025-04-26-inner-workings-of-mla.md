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

## Cache Size and Bandwidth

After computing the key and value vectors for a token, it makes sense to hang on to them—they’re needed for generating future tokens. So we allocate a large buffer to serve as a “KV cache” for storing these precomputed keys and values.

This KV-cache gets enormous for long sequence lengths--with DeepSeek-V3's dimensions and maximum supported sequence length of 128K tokens, standard Multihead Attention (MHA) would require a ~488GB KV-cache!

Given that, I had originally assumed that the issue was fitting this into GPU memory, but I've learned that the real bottleneck is **memory bandwidth**.

I'll use the statistics for an H100 to illustrate the point. It has:
* 80GB of off-chip GPU memory,
* 50MB of on-chip L2 cache,
* and can move data between them at 3.35 TB/s (the bandwidth).

(values taken from [here](https://datacrunch.io/blog/deepseek-v3-llm-nvidia-h200-gpu-inference-benchmarking))

The H100 can crunch 16-bit floats at just shy of a _petaflop_ (989.5 TFLOPs), but they need to be in that L2 cache first.

The core operation of attention scoring--multiplying a query vector with all of the key vectors for a sequence--can become highly **memory-bound**.

To attend to all of the tokens in the sequence, for each one we have to pull in 128 key vectors (one per head), with 128 floats each, a total of **16K floats** per token.

MLA reduces this dramatically--it only reads **576 floats** per token! That's about 28.44x fewer values to pull into the cache.

> Side note: That weird 576 value is the combination of two latents per token, one length 512 and the other length 64. We'll dive into both!

**Compute Cost**

Another interesting detail we'll see is that MLA requires **~4x more operations** to compute than standard MHA.

I think that highlights just how bad this bandwidth problem is--it's apparently a worthwhile tradeoff to make attention four times more expensive to compute in order to lower the bandwidth required!

As empirical evidence, MLA was introduced in DeepSeek-V2 ([here](https://arxiv.org/pdf/2405.04434)) and they report a higher maximum throughput for token generation, despite this added computation.


**Shorter Sequences**

That said, I'm gathering that attention _starts out_ as _compute_-bound, and for a given setup (model, GPU, batch size, ...) it only becomes memory-bound once you cross some particular sequence length.

I'm not sure if we can reasonably estimate this cross-over point given all of the things it depends on, but it seems worth highlighting that _MLA can be slower_ for "shorter" sequences, whatever that may mean.

**Implementation**

If you're interested in interpreting transformers and enjoy the algebra of attention, then MLA is a fascinating subject! Let's dig in.

## Trading Compute for Bandwidth

I think there are multiple conceptual framings that you could give to how MLA addresses this challenge.

I think the most straightforward is the following:

If we compute the key and value vectors and try to hang on to them to avoid recomputing, it does more harm than good because of bandwidth limitations. So... don't do it! Just store the token vectors instead.

This doesn't usually help in practice because, not only would we be recomputing the keys and values all the time, but the memory bandwidth is about the same because typically we set

```
head_dim  x  num_heads = embedding size
```

MLA solves this in two ways:

First, they project the tokens down to 512 dimensions, and cache these instead of the full embeddings (which are length 7K).

Second, they avoid re-computing the keys and values for every input... by not computing them at all.

You read that right--we don't actually need keys and values. Fundamentally, **each attention head only needs two projections**--one for the **input**, and one for the **output**.

In practice, we perform a (very) low-rank decomposition of a head's input projection into $W^Q_i$ and $W^K_i$ and a (very) low rank decomposition of the head's output projection into $W^V_i$ and $W^O_i$.

These decompositions are crucial for bottlenecking attention (forcing them to learn some structure) and reducing their compute cost (they're massive otherwise!).

This two-matrix view of attention is helpful here because it highlights that we have endless possibilities in how we choose to decompose and arrange the terms.

Let's look at how MLA chooses to break it down.


## Alternative Decompositions

Storing a compressed latent reduces the memory footprint of the cache dramatically, but only reduces the required memory bandwidth if we change how attention is calculated.

**Notation - Heads**

To understand the algebra of MLA, I think it's important to distinguish which projections are per-head vs. which are shared across heads. Wherever a matrix is unique to the heads I'll include the subscript $i$ to help clarify the distinction.

## From Query-Key to Pattern-Latent

Ultimately we are going to transform the attention calculation from a per-head query times per-head keys:

$a_i = q_iK^\top_i$


<img src='https://lh3.googleusercontent.com/d/1p7CtjkQOnqQTaPxMNJGHqHogmUTcRh1k' alt='Multihead perspective of calculating attention logits showing per head queries and per head keys' width='500'/>



Into a per-head "pattern vector" times **per-layer** latents:

$a_i = p_iC^{KV^\top}$

<img src='https://lh3.googleusercontent.com/d/1_XFLly-vAccjiRnY4kr1yvY4mtilDh0E' alt='Multihead perspective of calculating attention logits using per-head patterns and per-layer latents' width='500'/>


Note that the fact that there is only one set of latents doesn't mean there's less calculation, it just means that those latents are re-used by all of the heads.



**Re-formulating Attention**

Let’s start with a key insight from interpretability research (notably from Chris Olah and explained in Anthropic's Transformer Circuits Framework, [here](https://transformer-circuits.pub/2021/framework/index.html)).

The attention equations can be re-ordered to show that, for a given head $i$, the query and key matrices $W^Q_i$ and $W^K_i$ can be thought of as a low-rank decomposition of some larger matrix, $W^{QK}_i$


