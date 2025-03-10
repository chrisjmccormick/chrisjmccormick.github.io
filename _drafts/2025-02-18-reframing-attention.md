---
layout: post
title:  "Reframing Attention: A Matrix Decomposition Perspective"
date:   2025-02-18 17:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1fSOZ_RDAvFQTHbgg1SC3q1-_QSZdsfD_
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

The Attention equations from Vaswani et al. are a **reflection of computational efficiency**, optimized for **GPU execution**. However, if we step back from the implementation details, we can arrive at a cleaner, **more intuitive representation** of attention by reframing the equations.

One issue is that we often view Attention as a **monolithic process**, where we focus on the full multi-head representation instead of examining the behavior per head. The equations define per-head operations, yet our tendency is to pull back and look at large matrix multiplications over all heads at once. This obscures **key insights**.


## Per-Head Outputs


A major example: the **output projection matrix**. The standard attention equations present $W_O$ as a single learned transformation, but this hides the fact that **$W_O$ is actually a concatenation of per-head $W^O_i$ matrices**.

Once we make this distinction, we can rewrite the equations to show the **re-projected**
(model space) output of a **single head**.

For a single input (query) vector $x_q$, attending to a sequence of tokens in $X$:

<br/>

$\mathbf{o}_i = \alpha_i \,\bigl(X\,W^V_i\bigr)\,W^O_i \quad\in\;\mathbb{R}^{1\times d_{\text{model}}}$

<br/>


---

<img src='https://lh3.googleusercontent.com/d/1Y5jpuM_xOw58qoCYtXKItXPCYBNMv8Sz' alt='Single head re-projected output for a single input' width='250'/>

---

Previously, our equations and illustrations have only highlighted either:

* $z_i$ - The output of an attention head in the compressed **value space** $d_v$.
* $o$ - The **sum** of all head outputs, re-projected.



---

> _Side Note: I actually hit on this in a YouTube [video](https://youtu.be/kkJx_uarTmU?si=CxMkUgnj5g1cCyRI&t=1115) back in 2022 without realizing its significance!_
>
> _The video also captures the confusion that $W^O$ caused me--I explain it as taking a **weighted** combination of the heads._

---



## Separating Query-Key and Value-Output


Another issue is that we associate the **Value projection** ($W^V$) too closely with the **Query and Key projections** ($W^Q, W^K$).

Many illustrations show the query, key and value vectors all entering into an attention block together, and then the Output matrix applied as a final step over everything. This follows the original illustration in _Attention is All You Need_:


<img src='https://lh3.googleusercontent.com/d/1hnFPip3AwiOd44QbAxgZpfg4VqaR-JfP' alt='Original Attention is All You Need illustration' width='250'/>

This is again a reflection of how things are done optimally on the GPU, and obscures the fact that within each head, there are two independent processes going on:

1. The **Query-Key** process is calculating attention scores.
2. The **Value-Output** process is calculating updates to make to the input embedding.



On the GPU, we perform:

<br/>

$o = \text{Concat}\bigl(\alpha_i \,\bigl(X\,W^V_i\bigr)\bigr)\,W^O $

<br/>

But if we break apart the concatenation, and then apply the output projection _before_ the scores:

<br/>

$\mathbf{o}_i = \alpha_i \,\bigl(\bigl(X\,W^V_i\bigr)\,W^O_i\bigr) $

<br/>

The distinction between the two processes becomes much more clear.

<img src='https://lh3.googleusercontent.com/d/1AeHb7ZhG1pvGsVJHNGj3VDH-xb19up4f' alt='Separating the Query-Key and Value-Output processes' width='250'/>

What is the matrix that results from this Value-Output process, $ \bigl(\bigl(X\,W^V_i\bigr)\,W^O_i\bigr) $?

We've never noticed it. We don't have a name or a variable for it.

It's a matrix with size $T \times d_\text{model}$, and it contains the per-head, _per-token_, re-projected output.

Each row is a vector, in model-space, which carries the information that a token will contribute to the output of this attention head, if it is selected to do so by the Query-Key process. (i.e., it will be weighted by the token's attention score).

## Token Messages

There are interesting parallels between Graph Neural Networks and Transformers. Both models involve receiving information from context. In GNNs, a node receives an embedding from each node that its connected to, and these are referred to as **messages**. In a Transformer, tokens also receive embeddings from context, except globally (i.e., from every token in the sequence), and this label captures well that the "attention block" is how **tokens send and receive information**. 

> _Note: The language of GNNs has already had some adoption in Transformers, and GPT cued me into this. So far the term has only been used to describe the value vectors, not their more interpretable model-space representations. I think adopting the language here will really help with our mental models of Attention!_ 

For a given query vector, we have the messages matrix:

<br/>

$$
\mathbf{M}_i = \bigl(\bigl(X\,W^V_i\bigr)\,W^O_i\bigr) \quad\in\;\mathbb{R}^{T\times d_{\text{model}}}
$$

<br/>

Where each row is a message, $m_i$, from a token.



<img src='https://lh3.googleusercontent.com/d/1fSOZ_RDAvFQTHbgg1SC3q1-_QSZdsfD_' alt='The emergence of the Messages matrix' width='250'/>

For a given input vector, the output of attention head $i$ can now be elegantly viewed as the sum of the token messages, weighted by the attention scores.

$o_i = \alpha_i \,M_i $

where

* $\alpha_i \in \mathbb{R}^{1\times T}$  contains the attention scores for each token, and
* $M_i \in \mathbb{R}^{T\times d_{\text{model}}}$ are the tokens' messsages.

<br/>

The final output of the Attention block (including the residual connection) is the sum over the heads:

<br/>

$$
x^\prime = x + \sum_i{\alpha_i M_i}
$$

<br/>

---

> _Side Note: I think that the residual connection is critical for correctly understanding attention--it doesn't create a **new** vector to replace the input, it calculates an **adjustment** vector that's added to the input._

---


It's important to note that what we've defined here is a _conceptual_ implementation of Attention. It doesn't describe how Attention is calculated _in practice_, but rather defines a more intuitive explanation of what is happening.

Because of the order of operations we take on the GPU, these message vectors are never _explicitly calculated_ during Attention. However, we can choose to calculate them for the purpose of interpreting the model.


## Interpreting Messages


Uncovering these message vectors gives us a new way to **probe** the attention heads and learn more about their behavior.

**Messages are "Query Agnostic"**

A key insight from viewing Query-Key and Value-Output as two independent, parallel processes is that tokens _construct their messages without any knowledge of the attention scoring_.

In the Value-Output process, a token doesn't know what the current query is, so it is constructing the message it wants to communicate, through a particular head, _independent_ of whether it ends up selected by the scoring mechanism.

Instead of looking at what tokens are being attended to, we can come at an attention head from the opposite direction and see what each token _would_ send through that head if it got selected.



**Messages are in Model Space**

Furthermore, these messages are in model space, the same space as the input embedding and output embeddings.

The messages are a payload carried by the input embedding into the FFN and then onto the next layer.

We can potentially find the **message's recipient** by looking at:
1. Which **neurons** in the FFN the message aligns with.
2. Which **heads** in the **next layer** the message aligns with.

In order to perform the second analysis, we need to uncover the correllary to the "messages" in the Query-Key process--what I'm referring to as the head "patterns".


## Head Patterns

In the same way that we've uncovered a per-token "message" in model space by re-arranging the order of operations, we can also compute a per-head "pattern".

To calculate our attention scores, we normally:

1. Project our tokens onto query and key space.
2. Multiply the query against the keys to get the attention logits.
3. Scale down, and normalize the result with the softmax function.

For a single query vector $x_q$ and a sequence of tokens $X$, this can be expressed as:


$$
\alpha_i = \mathrm{softmax}\!\Bigl(\frac{(x_q \, W^Q_i)\,{\bigl(X \, W^K_i\bigr)}^\top}{\sqrt{d_k}}\Bigr),
$$

where  $ \alpha_i \in \mathbb{R}^{1 \times T} $ holds the attention score between the query and every token.

As before, we can change the order of operations in this equation to produce a unique and valuable interpretation.

Instead of producing queries and keys, we can multiply the two projection matrices first, and then calculate the attention logits as:

<br/>

$x_q(W^Q_i {W^K_i}^\top){X}^\top$

<br/>

Our pattern vector comes from the first three terms:

<br/>

$p_i = x_q(W^Q_i {W^K_i}^\top)$

<br/>

Now, the attention scores are based on the dot product between this "pattern" vector and each of the token embeddings:

<br/>

$$
\alpha_i = \mathrm{softmax}\!\left(\frac{p_i X^\top}{\sqrt{d_k}}\right)
$$

<br/>

This ordering of operations isn't efficient computationally--it increases the number of operations by a factor of $\frac{d_{model}}{2d_v}$, requiring more memory and more compute.

But as with the messages, it's something we can choose to compute for the value of interpretability.


## Interpreting Head Patterns

Attention scores are based on how well a token embedding aligns with a pattern vector (i.e., how large their dot product is).

For an input vector $x_q$ and a particular head $i$, we can think of $p_i$ as the pattern that the head is looking for among the tokens in the sequence.

That's the same description that we'd give to the existing approach of comparing a query to the keys, but with a key distinction--the $p_i$ vectors are in **model space**.

This means that we can compare these head patterns to:

1. The **vocabulary embeddings**: We can search the vocabulary to identify any and all tokens that this pattern would align with well. 
2. The **position encodings**: We can easily see which positions the head is biased towards or against. 
3. The **messages**: Multiply every token's message with every head's pattern to see how the tokens in one layer communicate with the heads in the next.
4. The **output neurons**: Which FFN outputs is the head sensitive to?

I've explored the first 2 concepts in a later post with interesting results--I'm excited to explore 3 and 4!

---

> _Side Note: Mapping the messages may be more complicated because of the FFN's contribution. Perhaps sending the input embedding through the FFN with just a single message attached to it will help us isolate the mappings?_

---

## Singular Value Analysis

In the next section we'll look at another exciting revelation that arises from this "conceptual" version of the Attention equations: merging $W^QW^K$ to form $W^P$, the projection matrix for the patterns, and merging $W^VW^O$ to form $W^M$, the projection matrix for the messages.

If we construct those larger matrices and then apply Singular Value Decomposition (SVD) we can analyze their top singular vectors, which could serve as another tool for interpretability. 

I haven't explored this yet, but GPT seems to like the idea, so I'll share its thoughts for now:

---

* Top Singular Vectors in $W^P$ might reveal dominant directions that might correspond to specific token categories or syntactic roles.  
* Top Singular Vectors in $W^M$ might uncover latent dimensions that inform how the model communicates or transforms information internally.  
    * This could further our understanding of how “messages” are encoded and whether they align with semantic or syntactic distinctions.  
* Inverting $W^P$--"Reverse Mapping":
    * Inverting $W^P$ (or using a pseudo-inverse) might allow you to map pattern vectors back into the original embedding space.  
    * This could shed light on how well the projection preserves information and might reveal hidden correlations or transformations that are not immediately obvious.

---


# A Matrix Decomposition Perspective on Attention


The inspiration for this decomposition perspective came from my recent deep dives into LoRA and Multihead Latent Attention (MLA).

It stood out most clearly first on the Value-Output side of attention, so I'll start there.


## The Merged Value-Output Matrix: $W^M$




In the equation for the re-projected head output:

<br/>

$\mathbf{o}_i = \alpha_i \,X\,W^V_i\,W^O_i $

<br/>

it becomes more apparent that $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and
$W^O_i \in \mathbb{R}^{d_v \times d_{\text{model}}}$ can be viewed as the decomposed version of a larger matrix with rank $\le d_v$.

Conceptually, this larger matrix is what performs the **transformation** of our input vectors into **messages**, so we'll name it accordingly.

<br/>

$W^M_i = W^V_i W^O_i$

<br/>

For a given head $i$, the messages for all of the tokens could be computed as:

<br/>

$M_i = XW^M_i$

<br/>


## The Merged Query-Key Matrix: $W^{P}$


In the previous post, I noted how we can re-order the calculation of the attention logits from comparing query and keys:

<br/>

$(x_q \, W^Q_i)\,{\bigl(X \, W^K_i\bigr)}^\top$


  ⬇



$x_q(W^Q_i {W^K_i}^\top){X}^\top$

<br/>

The result of that central matrix multiplication is our **pattern projection** matrix, 

<br/>

$$
W^P_i = W^Q_i {W^K_i}^\top
$$

<br/>

For a given input vector $x_q$ and head $i$, the head's pattern vector can be calculated as:

$p_i = x_qW^P_i$


Calculating the messages this way, though, would be inefficient. We've increased the number of parameters in the Value-Output process dramatically
(by a factor of $\frac{d_{model}}{2d_v}$), requiring more memory and more compute.

But they open up interesting opportunities for interpretation!

# Conclusion

I think we could call this the Fused model of Attention, because it describes the conceptual model of Attention where the query-key and value-output matrices have been fused into patterns and messages.. However... I think a common mistake is to name developments in a way that references what came before.

I think there is an opportunity to move forward here and find better expressions. There is additional valuable terminology to be borrowed from GNNs--for instance, "attention" merely describes the scoring process, a node also "Aggregates" its messages. Perhaps the overall block becomes something more like the message "Aggregator"?

