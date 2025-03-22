---
layout: post
title:  "Patterns and Messages - Part 3 - Alternative Decompositions"
date:   2025-02-19 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1yx5AuQ1P9uF9yqim7Em4EQXmwXnwmN46
tags: Transformers, Attention Mechanism, Low-Rank Decomposition, Model Compression, GQA, Joint KV Compression, DeepSeek, Transformer Optimization, Token Efficiency, Mechanistic Interpretability, KV Cache, MultiHead Latent Attention, MLA
---

One potential benefit for this merged perspective is that it lets us begin our research into Transformer efficiency from a "more fundamental" definition of Attention.

We know we have (at least) three issues to resolve:

1. **Compute Cost** - $W^P_i$ and $W^M_i$ are expensive to project on to,
2. **Cache Size** - We want to cache the calculated patterns and messages, but they're very large, and this becomes a substantial problem with longer sequence lengths.
3. **Low Rank** - We also believe that $W^P_i$ and $W^M_i$ are over-parameterized--that the heads should operate on far fewer features than the full embedding space.

Our standard query-key, value-output decomposition is one approach for addressing all three, but there might be alternatives.

_by Chris McCormick_


## Contents

* TOC
{:toc}

## Message Tying

My understanding is that, while sometimes we may see a head focusing in on a few particular related words, the attention pattern is often more dispersed than that. Whatever information is being shared is coming from a composition of the token messages.

Could an attention head leverage multiple compositions of the same set of messages?

This is **Group Query Attention (GQA)**, which is used in the Llama models, among others.

Using the language of GQA, Llama 3 8B has 32 query heads in groups of four, and 8 key-value heads.

It may be a little easier to think about when you frame that as 32 pattern heads and 8 message heads.

From a token perspective:

* Each token gets to write 8 messages.
* For each message, it creates four different patterns--four different criteria for matching.

From an overall attention block perspective, the below illustration might help.

A layer has 8 message projections, $W^M_{1-8}$ which produce 8 sets of messages for the sequence, $M_{1-8}$. I've illustrated $M_1$ and $M_2$ below.

A layer also has 32 pattern projections, $W^P_{1-32}$, which produce 32 sets of patterns for the sequence, $P_{1-32}$, and I've shown $P_{1-8}$.

<img src='https://lh3.googleusercontent.com/d/1EbKFq5BFeb9dS68LSA4oJnE_JTp7wwfP' alt='Illustration of Group Query Attention showing 8 unique sets of Patterns and 2 sets of messages, replicated four times each' width='350'/>

The outputs $o_{1-4}$ are four different compositions of the same set of messages in $M_1$, and so on.

It seems difficult to keep straight no matter how you explain it, but perhaps it helps to reduce the number of projections we need to think through.

The outcome and main purpose of GQA is the reduction in cache size. Instead of caching 32 sets of patterns and messages (stored in practice as their keys and values), we are storing 1/4 as many messages.

## Matching a Pattern to Tokens

I found it conceptually most helpful to frame Attention as producing a pattern vector to go with each message vector. This is also consistent with how we implement things--storing a key and a value, per token, per head.

However, if you were to actually implement Attention as patterns and messages, there's an obvious switch to make there for efficiency.

For a given token, instead of storing a pattern vector for each head, we can just store the word vector itself!

Let's build on GQA to demonstrate this. In the below illustration, instead of storing the token patterns for all 32 heads, we are just storing the tokens, which can be used for attention across all 32 heads.

(Note: I didn't include input vector, $x_q$ in this version. $x_q$ is projected onto each of the $W^P_{1-32}$ matrices to produce the patterns $p_{1-32}$ shown.)

<img src='https://lh3.googleusercontent.com/d/17T5bO-lKY2hVhZfKTWIsrXQu6qOb1EXx' alt='Group Query Attention, except we are storing the tokens instead of the patterns' width='350'/>

It's mathematically equivalent, and a massive reduction in cache size!

_However_, we've only addressed the **cache size** problem, and not **compute cost** or the **low rank** nature of the heads.

Standard attention focuses on decomposing the projections to solve this, but what if we flipped the problem and instead tried to make the tokens themselves smaller? That's the idea behind token compression.

## Compressed Tokens

Another way to make the pattern and message projections smaller would be to make the tokens smaller. 

For the current input / "query" token $x_q$, we could try learning a compressed representation:



<br/>

$x_q^C = x_qW^C_i$

<br/>

Where $W^C_i$ compresses the token down to the head size (e.g., from 4,096 down to 128).




<img src='https://lh3.googleusercontent.com/d/1Tevs-yXpYhn0K5SnTS_Zq_v-sGwnoBIL' alt='Compressing the input token to cache it' width='400'/>


It's starting to sound a lot like regular query-key Attention, but the key difference here is that these compressed tokens can be used for both the patterns _and_ the messages:  

<img src='https://lh3.googleusercontent.com/d/1yx5AuQ1P9uF9yqim7Em4EQXmwXnwmN46' alt='Joint KV Compression illustrated instead as learning a token compression, with the pattern and message projections still in place.' width='900'/>

This technique is called "Joint KV Compression", and the $W^C$ matrix might be called the "KV-down" matrix, $W^{KVD}$. 

This cuts the cache size **in half** compared to standard attention--we only store 1 embedding per token, per head, instead of 2.

I think the pattern-message framing is especially nice here. 

Instead of:
* queries
* kv latents
* output projection

We have:
* patterns
* compressed tokens
* message projection


> Note: In the illustration, I've gone back to the standard approach of scoring the tokens _prior_ to the message projection.  
>
> You could show the messages here, but I didn't want to imply that the messages are still cached.



**Grouping**

There's no longer a cache-size benefit here to reducing the number of message heads as we do in GQA. 

However, perhaps the grouping concept is still relevant, and every four heads could share the same token compression, so we'd have $W^P_{1-32}$, $W^C_{1-8}$, and $W^M_{1-32}$.

This would reduce the cache size from `num_tokens * 2` in standard attention down to `num_tokens / 4`. 

I haven't researched to what degree KV compression has been explored; however, I know that it's been used successfully by DeepSeek in DeepSeek-V3, with a further modification: a single learned token compression for **all heads** in a layer.

## Per-Layer Token Compression





Remarkably, DeepSeek demonstrated in their "MultiHead Latent Attention" (MLA) architecure, used in DeepSeek-V2/V3/R1, that it's possible to learn a single token compression per layer (instead of per head).

This gets us back to something like my earlier illustration, of storing just one copy of the tokens for all heads:

<img src='https://lh3.googleusercontent.com/d/17T5bO-lKY2hVhZfKTWIsrXQu6qOb1EXx' alt='Group Query Attention, except we are storing the tokens instead of the patterns' width='180'/>

This is mathematically equivalent, so it's perfectly reasonable to think that we could use just one version of the tokens for all heads.

However, I'd expect that if we compressed the tokens all the way down to length 128 and tried to use this representation for _all_ of the heads in a layer, that there wouldn't be enough information retained to provide unique functionality per head. 

Apparently **512** dimensions is enough, though! (This is the length used by DeepSeek-V3)

We can learn a single compression matrix $W^C$ per layer which takes the tokens down to length 512:

<img src='https://lh3.googleusercontent.com/d/1caUOskYY7mm8BpjIPj6hpktwOPO6PoBY' alt='A single learned token compression per layer, $W^C$, allowing us to store just one compressed copy of the tokens per layer in the cache.' width='400'/>

There are a few issues to address, though, before we can simply swap in these length 512 vectors:

1. There is a **computation cost** to this--we'll have to perform the attention calculation on length 512 vectors instead of length 128. That means attention will require **4x more** multiply-accumulate operations.
2. We still have the low-rank problem--512 dimensions is still too many.
3. We need to address how to handle encoding position with **RoPE**.



**Compute Cost**

I was able confirm from the DeepSeek-V3 code that they simply eat this cost--they perform the per-head attention calculations on length 512 vectors. No workarounds there.



**Low-Rank Heads**

MLA solves the rank issue in the standard way, by decomposing the Pattern and Message projections to create a bottleneck of 128 features.

Below is the updated diagram reflecting these changes.

Note the following:
1. The (compressed) tokens are now length 512, and don't have a per-head subscript.
2. The pattern and message projections have increased in size to 4k x 512.


<img src='https://lh3.googleusercontent.com/d/15Bn5tzdHcDA5w0VoJCSUaGjfvQMO10e6' alt='The beginnings of MultiHead Latent Attention. This illustration shows joint KV compression to a larger size, but only one compression matrix per layer.' width='900'/>


Also, I didn't draw the Pattern and Message decompositions, but wrote their dimensions below each matrix. These reduce the rank of these projections to 128 (and reduce the compute cost as well!).


**Pattern and Message Framing**

I think that the Pattern and Message framing is _especially_ valuable here. In fact, MLA was what inspired me to explore the matrix merging concept.

Part of the joint KV compression strategy is to learn a shared down projection, $W^{KVD}$ with size `[4k x 512]` (this is our $W^C$ matrix), but then separate "Up" projections for the keys and values, $W^{KU}$ and $W^{VU}$. 

"Up" is a bit of a misnomer here, though, because really we are further projecting down to the head size of 128: 

$W^{KU}_i$ and $W^{VU}_i$ are `[512 x 128]` each.

But here's the fun part--they re-arrange the equation so that they end up calculating $W^Q_iW^{KU'}_i$ and $W^{VU}_iW^{O}_i$... _the patterns and messages!_

<br/>

$W^P_i = W^Q_iW^{KU'}_i$

$W^M_i = W^{VU}_iW^{O}_i$

<br/>

The right hand side of those two equations correspond to the decompositions I noted in the illustration.

This re-arranging is crucial--it's only by working with the Patterns and Messages that you can get away with just storing the tokens. Otherwise, you would just be re-computing the keys and values at every step.



**RoPE**

Unlike the positional encoding vectors of early models, which were added on to the initial embeddings, RoPE is applied during the Attention process at every layer of the model.

To work properly, it needs to be applied separately to the input token and the context tokens. In standard attention, we:

1. Project the queries and keys
2. Apply RoPE to both
3. Multiply queries with keys to get attention scores.

I'm curious whether it would work / whether DeepSeek considered applying RoPE to the length 512 patterns and compressed tokens.

Instead, they saw it as necessary to apply RoPE to the 128 dimensional queries and keys. This creates a problem, because we've re-arranged the terms such that the keys never exist!

I'll save the detailed explanation for a dedicated post on MLA, but they essentially solved this by creating a **parrallel set of attention heads** with length 64 specifically for handling position information. 

These RoPE heads follow the query-key process rather than pattern-token, and result in calculating an entire second set of attention logits! 

The attention logits from the RoPE heads are added to the logits from the regular heads, and then SoftMax is applied. 

The total compute cost is essentially the same as if the heads were length 576 (512 + 64).

**Minutiae**

Since this is _so close_ to a complete description of MLA in DeepSeek-V3, I'll go ahead and mention two remaining details:
* The V3 embedding size is 7k.
* They insert an additional "down" step in calculating the query vector: 7k --> 1.5k --> 128. 

MLA is quite an interesting architecture overall (compressing down to 512 is fascinating from an interpretability perspective!), and I think the pattern-message framing really adds insight to it. 

## On Pre-Trained Models

It's possible to take a pre-trained LLM and merge the weight matrices to construct these Pattern and Message projection matrices.

This is useful for probing them to see what we can learn about how the model works.

However, if the matrix merge is an uncommon perspective on Attention, I'm curious whether there might be some techniques that haven't been considered for improving the efficiency of existing pre-trained LLMs.

If we **merge** the Query-Key and Value-Output matrices for each head, we can then look for opportunities to **decompose** them again **more effectively**.

I had a couple ideas to explore around this which I'll discuss in the "Further Research" post at the end of the series, but briefly:

We could try forming the $W^P_i$ and $W^M_i$ matrices, and then:

1. Decompose them with SVD back into QKVO, but keeping only the top singular values.

Or

2. Find a joint decomposition, to create the architecture I described in the 'Compressed Tokens' section. 

These may not work due to RoPE, however.





## Conclusion

By providing a more "foundational" explanation of Attention, I hope that the patterns and messages framing will continue to provide insight into these attention methods, and maybe inspire some creative new approaches!

In the next post, we'll look at another fresh perspective on Attention--that we can view each head as a dynamically created neural network, where the patterns and messages serve as its input and output weights.
