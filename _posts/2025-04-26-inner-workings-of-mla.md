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



### Pattern Projection


I like to write this conceptual larger matrix as:

$W^P_i = W^Q_iW^{K^T}_i$

Where 'P' stands for "pattern", because this matrix is used to project a kind of template vector to be matched against all of the tokens in the sequence.


<img src='https://lh3.googleusercontent.com/d/18fiJf-r1QIrSkHcbyaiNCYNLceKIdqN2' alt='Merging the query and key projection matrices into the pattern projection' width='400'/>



We can write the formula for attention very cleanly with this matrix.

For each head, the input vector (the query) is projected to produce its pattern vector:

$p_i = xW^P_i$

The attention logits for head $i$ are simply the dot product between this and all of the token vectors in the sequence:

$a_i = p_iX^T$

This perspective provides a more "fundamental", _conceptual_ definition of attention that's very useful here because it highlights:

1. That our "Query and Key" matrices are **just one way** to decompose attention--we'll see that MLA does it differently.
2. That **we don't have to project both sides** of the equation--it's enough to project _just the input vector_ and compare that to the raw token vectors.

The second insight is powerful because it allows us to leverage **broadcasting** to do the same math with **fewer memory reads**.

Attention is calculated separately for each head, but we get to use the same token vectors across all heads:

$a_1 = p_1X^T, \quad a_2 = p_2X^T, \quad ..., \quad a_{128} = p_{128}X^T$

i.e., we'll produce a per-head pattern vector, but then broadcast these across the sequence.

In contrast, standard attention requires producing per-head representations on both sides:

$a_1 = q_1K_1^T, \quad a_2 = q_2K_2^T, \quad ..., \quad a_{128} = q_{128}K_{128}^T$

This requires reading in a separate set of keys for every head.





**High Compute Requirement**

This is a neat trick, but there's a big problem here that you may have noticed.

That $W^P_i$ matrix is _huge_. It's 7K $\times$ 7K, and there is one per head.

The patterns are the same size as the embeddings, so instead of multiplying a length 128 query with a length 128 key, we'd be multiplying a length 7K pattern with a length 7K token vector!

The two approaches are mathematically equivalent--they produce the exact same attention score--but the pattern formulation requires far more operations.

MLA solves this by _compressing the input vectors prior to attention_.


### Compressions

The size of the pattern matrix is dictated by the size of the input vectors; we can shrink it by shrinking the inputs.

In standard attention we'd calculate the attention logits for head $i$ as:

$a_i = xW^P_iX^\top$

Where,

| Symbol       | Shape                  | Description                                               |
|--------------|------------------------|-----------------------------------------------------------|
| $x$          | 1 $\times$7K   | Input vector for query   |                          |
| $W^P_i$      | 7K $\times$7K        | Head $i$'s pattern projection matrix                      |
| $X^\top$     | 7K $\times$ n         | Sequence vectors for keys    |
| $a_i$        | 1 $\times$ n           | Head $i$’s attention pattern over the sequence            |

But MLA first compresses the inputs (to different lengths!) so that we have:

$a_i = c^QW^P_iC^{KV^{\top}}$

Where,

| Symbol       | Shape                  | Description                                               |
|--------------|------------------------|-----------------------------------------------------------|
| $c^Q$          | 1 $\times$ 1.5K   | Compressed input vector for query   |                          |
| $W^P_i$      | 1.5K $\times$ 512        | Head $i$'s pattern projection matrix                      |
| $C^{KV^{\top}}$     | 512 $\times$ n         | Compressed sequence vectors for keys (and values)   |
| $a_i$        | 1 $\times$ n           | Head $i$’s attention pattern over the sequence            |

<br/>

> Note that, in both cases, we will (further) decompose $W^P_i$ into $W^Q_iW^{K^{\top}}_i$ with an inner dimension of 128--we'll do this in the next section.


**Compression Matrices**

These compressed vectors are created from two learned matrices, which are shared by all heads in a layer:

<br/>

| Symbol         | Shape                  | Description                                               |
|----------------|------------------------|-----------------------------------------------------------|
| $W^{DQ}$       | 7K $\times$ 1.5K | Compression matrix for the **query** input               |
| $W^{DKV}$      | 7K $\times$ 512  | Shared compression matrix for the **key** and **value** inputs |

<br/>

These produce the compressed representations we saw above:

<br/>

| Symbol         | Shape                  | Description                                               |
|----------------|------------------------|-----------------------------------------------------------|
| $c^Q$          | 1 $\times$ 1.5K  | Compressed input vector for query: $xW^{DQ}$             |
| $C^{KV}$       | n $\times$ 512   | Compressed sequence vectors for keys and values: $XW^{DKV}$  |

<br/>

The below illustration shows these compressions.



<img src='https://lh3.googleusercontent.com/d/1I1xH9FJBPajsmwSgV2kUF4deW6RhztrI' alt='Illustration of the two compression matrices, one for the query and one for the shared key value latent' width='900'/>



The key-value latents are stored in the KV cache to be re-used as we continue to generate new tokens.

But note the massive difference there!

In standard attention, each layer of the DeepSeek model would produce and cache 128 key vectors and 128 value vectors per token, each of length 128. That's 32K floats total.

In contrast, MLA stores a single length 512 latent per token.

> Again, that yields a 64x smaller footprint, but note that the bandwidth savings are only half that, since the latents must be read twice.



**Side Note: Interpretability**

This is a fascinating quality of MLA to me, from an interpretability perspective. (Maybe skip this bit if you're not versed in that field?)

Each key and value head is allowed to project its own 128-dimensional subspace to read from / write to, but they are all constrained to operate within the same 512-dimensional subspace of the residual stream.

Could that maybe force the heads in a given layer to have a more homogenous set of functions? I'd love to dig into that!

**Decompression Step?**

Initially, I had assumed that the model would need to be trained with a decompression step, $W^{UKV}$ with shape 512 $\times$ 7K, in order to learn this "compression" behavior, but that's not the case. (This lead to a lot of confusion on my part, unfortunately!)

You could think of $W^{UKV}$ as being folded into the key and value projections, or just dismiss the notion entirely.

> Side note: The authors chose to rename the QKV projection matrices to each include a "U"--e.g., $W^{UK}$--and I think that's partially what lead to my confusion. Conceptually, the QKV matrices are further _down_ projections to the 128 dimensional heads.




### Decomposition into Query-Key

We still want to decompose the pattern projection, as usual, into a $W^Q_iW^{K^\top}_i$ with a small inner dimension (the head size, 128).

Let's first look at the creation of the pattern vectors, since this done as a separate step before computing attention.

Head $i$ extracts a "template vector" / pattern $p_i$ to match against the sequence tokens:

$p_i = c^Q \cdot W^{UQ}_i \cdot W^{UK}_i$

Where,

| Symbol                 | Shape                        | Description                                                        |
|------------------------|-------------------------------|--------------------------------------------------------------------|
| $c^Q$                  | 1 $\times$ 1.5K         | Compressed input vector for the query                              |
| $W^{UQ}_i$             | 1.5K $\times$ 128       | Query projection matrix for head $i$                               |
| $W^{UK^\top}_i$             | 128 $\times$ 512       | Key projection matrix for head $i$                                 |
| $p_i$                  | 1 $\times$ 512            | Head $i$’s pattern vector for the input token   |

The 128-dimensional head size makes the pattern projection cheaper to compute, but perhaps more importantly it "bottlenecks" the attention head to avoid it being over-parameterized / add sparsity / learn to specialize / pick your favorite interpretation.

The below illustration captures this step.


<img src='https://lh3.googleusercontent.com/d/1TkaHaLIG31pjUKYizLssDhnT_V364lJt' alt='Illustration of the path from query latent to pattern vector' width='900'/>



### Attention Scoring




So the attention scores for head $i$ are now:

$a_i = p_i \cdot C^{KV^\top}$

This is shown in the below illustration--for convenience, I've ignored proper matrix orientation. Think of $C^{KV}$ as a table of latents, one per row. Take the dot product between the pattern vector and a latent to produce the corresponding logit.


<img src='https://lh3.googleusercontent.com/d/1zBmmMz_X7YmyGDX3W9doWfqh5rILQhJy' alt='Illustration of calculating the attention logits by multiplying the pattern vector with the sequence latents' width='500'/>


This is the multiplication of a 512-dim pattern vector with 512-dim token latents--far fewer operations than 7K times 7K, but still _4x_ higher than our usual queries-times-keys operation (done at 128-dims).

That's quite the cost to pay, but it's considered a worthwhile trade-off due to the savings on **memory bandwidth**--the attention calculation is often _memory-bound_ rather than compute-bound.


**Multihead View**

A multihead illustration is helpful, I think, for highlighting the difference between MLA and MHA.

In the below to illustrations, we are calculating attention between **one** input token and the full sequence of **n** tokens.


_Multihead Latent Attention_

<img src='https://lh3.googleusercontent.com/d/1_XFLly-vAccjiRnY4kr1yvY4mtilDh0E' alt='Multihead perspective of calculating attention logits using per-head patterns and per-layer latents' width='500'/>


_Multihead Attention_

<img src='https://lh3.googleusercontent.com/d/1p7CtjkQOnqQTaPxMNJGHqHogmUTcRh1k' alt='Multihead perspective of calculating attention logits showing per head queries and per head keys' width='500'/>





Note, _critically_, that there is a per-head pattern vector, but only a single set of sequence latents.

For each token, instead of reading 128-dim keys from 128 heads (16k values total) we're just reading a 512-dim latent. That's **32x less memory reads**  for calculating the attention scores.

We can achieve similar gains on the output side of attention as well.

## Continued in Colab...

I'm struggling with MathJax equation rendering issues. Google Colab is apparently much more forgiving, and is where I drafted this post originally. 

Until I've fixed the issues, please see the Colab Notebook [here](https://colab.research.google.com/drive/1JC1cEsk-3SxIUPWL7wF0eveelyo3e7fy#scrollTo=a2ubJZzjuI7V) to read the remainder of the post. Sorry for the inconvenience!
