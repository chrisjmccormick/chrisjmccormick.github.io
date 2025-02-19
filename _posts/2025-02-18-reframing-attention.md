---
layout: post
title:  "Reframing Attention: A Matrix Decomposition Perspective"
date:   2025-02-18 17:00:00 -0800
comments: true
image:
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

The Attention equations from Vaswani et al. are a **reflection of computational efficiency**, optimized for **GPU execution**. However, if we step back from the implementation details, we can arrive at a cleaner, **more intuitive representation** of attention by reframing the equations.

One issue is that we often view Attention as a **monolithic process**, where we focus on the full multi-head representation instead of examining the behavior per head. The equations define per-head operations, yet our tendency is to pull back and look at large matrix multiplications over all heads at once. This obscures **key insights**.


### Per-Head Outputs

A major example: the **output projection matrix**. The standard attention equations present $W_O$ as a single learned transformation, but this hides the fact that **$W_O$ is actually a concatenation of per-head $W^O_i$ matrices**. 

Once we make this distinction, we can rewrite the equations more simply in terms of the behavior of a **single head**, revealing new insights into how attention operates.


$$
O_i = a_i (X W^V_i) W^O_i
$$

(TODO)

### Separating Query-Key and Value-Output Transformations

Another issue is that we associate the **Value projection** ($W^V$) too closely with the **Query and Key projections** ($W^Q, W^K$). 

Illustrations typically show the query, key and value vectors all entering into an attention block together, and then the Output matrix applied as a final step over everything.

This is again a reflection of how things are done optimally on the GPU, and obscures the fact that within each head, there are two independent processes going on:

1. The Query-Key process is calculating attention scores.
2. The Value-Output process is calculating updates to make to the input embedding. 

The scores from (1) are then used to take a weighted average of the update vectors from (2), and this produces a single vector which is the output of an attention head.

By representing $W_O$ as a single matrix which is multiplied over all of the concatenated heads, it's easy to misinterpret this as taking a weighted combination of the head outputs. 

In fact, the output of each head is a vector in _model space_, and these are summed together and added to the input embedding to create the final output of MHA.


Side Note: A further source of confusion is the equating of $d_k$ (the key/query dimension) to $d_v$ (the value dimension). These are **independent**, but since models typically set them equal ($d_v = d_k$), they are collapsed into a single "head size" parameter.


## Inspiration from MLA


DeepSeek-V2 introduced **Multi-Head Latent Attention (MLA)**, where they **explicitly decomposed** the Query, Key, and Value projection matrices.

Instead of projecting directly, they introduce a compressed latent space:

- A shared projection **$W^{KVD}$** compresses the input into a latent space of dimension 512.
    - TODO - Is this per-head? I don't think so.
- A **$W^{KU}_i$** and **$W^{VU}_i$** which project the compressed latent representation "Up" into Key and Value space.
    - (I put "Up" in quotes because the per-head Key and Value dimensions are actually smaller, 128, so it's more like "Down-Down")

This is primarily done to **reduce KV cache size**—a crucial optimization for large models.

But here’s where it gets interesting: they realized that, through basic algebra, they could **fold** these projections into **$W^O$ and $W^Q$**. That is:

- **$W^{VU}$ merges into $W^O$**
- **$W^{KU}$ merges into $W^Q$**

This is similar to **LoRA**, where we train with a decomposition and then fold the trained low-rank matrices back into the main weight matrix for inference.


## A Matrix Decomposition Perspective on Attention


It shocked me that they could simply merge these operations away. And yet, thinking about it--$W^Q$ and $W^K$ are just linear transformations, and $W^V$ and $W^O$ are also just linear transformations. 

They need to be learned separately, but once the model has been trained, they certainly _could_ be collapsed for inference.

We don't do this in practice because it increases the amount of computation required to evaluate attention, but it does lead to some interesting perspectives.  


### **1. The Merged Query-Key Matrix: $W^{P}$**


For a single head, we typically think in terms of first projecting the tokens, $X$ onto Query space and Key space:

$Q_i = X W^Q_i$

$K_i = X W^K_i$ 

And then computing the logits as 

$$
\text{Attention logits} = Q_i K_i^T
$$

However, we can change the order of operations in this equation to produce a different interpretation.

First, collapse the two projection matrices into a single larger one, 

$$
W^P_i = W^Q_i (W^K_i)^T
$$

Then, project the token embeddings:

$$
P_i = X W^P_i
$$

And finally calculate attention logits:

$$
\text{Attention logits} = P_i X^T
$$


**1. Low Rank Interpretation**

This reframes the query and key matrices as the decomposition of a larger matrix with rank $\le d_k$.   


**2. Model Space**

The embeddings in $P_i$ are in model space. This is significant for interpretability, because "model space" is the one that we can interpret--it ties to the token embeddings. 
Words are points in this space, and the distance between them reflects semantics. We can compare vectors with cosine similarity, and do arithmetic on them like the classic (queen - king) + prince = ?  

I demonstrate the value of this with experimental results further down.

---

If we write self-attention in terms of per-head operations, we see that attention **logits** are computed as:

$$
\text{Attention logits} = (X W^Q_i) (X W^K_i)^T
$$

This can be refactored such that we first compute 

But we can rewrite this in a form that **reveals an implicit low-rank structure**:

$$
X W^{QK}_i X^T, \quad \text{where } W^{QK}_i = W^Q_i (W^K_i)^T
$$

This means that **each head applies a rank-limited transformation to model space**, computing a new representation before scoring interactions between tokens.



### **2. The Merged Value-Output Matrix: $W^M$**


Once we've broken up $W_O$ into heads, the output of a head can be expressed as: 

$$
O_i = a_i (X W^V_i) W^O_i
$$

Here again we can **collapse** $ W^V $ and $ W^O $ into a single transformation:

$$
W^M_i = W^V_i W^O_i
$$

And the output becomes

$$
O_i = a_i (X W^M_i)
$$

TODO, GPT's take: This tells us that each head’s behavior can be fully described as a **low-rank transformation on input embeddings**, with rank at most $ d_v $.

The final output of attention is the straight sum across the heads.

$ o_t = x_t + \sum{o_i,t} $

TODO


## **Experiment: Investigating Head Patterns**


Here's why I chose the letters $P$ and $M$.

Because they are in model space, I believe they have clear interpretations which can be demonstrated by experiment.

**Query-Key --> Patterns**

When we project an input embedding $x$ onto $W^P_i$, we are extracting a **pattern**, $p_i$, in model space, which **the head is looking for**.

We can infer this because the attention scores are calculated as the dot product between $p_i$ and all of the embeddings in the sequence, $X$. If the pattern is present in a particular token embedding, the dot product with $p_i$ will be high and the score will be high. 

**Value-Output --> Modifiers**

In the Value-Output process, each head outputs a matrix of "**modifier**" vectors, $M_i$. (With dimensions $ T \times d_\text{model}$) 

Each of these vectors $m_\text{i,t}$ represents that token's proposed modification to the input. A head's final contribution to the attention output is a weighted average of these modifier vectors. The weights are the attention scores.


## Experiment: Comparing Attention "Patterns" and "Modifiers" to the Vocabulary 



I ran an experiment where I extracted the **pattern vectors** $p = x W^{P}_i$ for a few tokens and a number of different heads, and then compared these patterns to the vocabulary embeddings via cosine similarity.


* TODO - Next, extracted the modifier vectors for those words, and compared these to the vocabulary. 

Here are some interesting results from BERT-base:


**Layer 0, Head 3, Word = "happy"**



_Search Pattern_

The pattern vector extracted by head 3, layer 0 is closest to:
  - make, making, made, makes, people

This tells us that for the query word "happy", this head is "searching for" that second list of words. 




_Head Modification_

This suggests that if we extract the modifier vector for "make" via $ m = xW^M_i $ we might see something interesting... (TODO) 



## **Why This Matters**

This reframing has several implications:

- **Reveals rank constraints**: Each head operates with rank at most $d_v$, which may limit how information is represented.
- **New ways to regularize attention**: Understanding per-head transformations could inspire better weight constraints or alternative decompositions.
- **Better interpretability**: Instead of treating attention as a black box, we can analyze how each head rewrites meaning.




## **Final Thoughts**

This reframing doesn’t change how Transformers are implemented, but it **changes how we think about them**. Rather than seeing multi-head attention as **concatenation followed by an output projection**, we can view it as **a sum of independent, low-rank modifications to model space**.

Could enforcing sparsity or alternative decompositions improve future architectures? There's a lot to explore.

