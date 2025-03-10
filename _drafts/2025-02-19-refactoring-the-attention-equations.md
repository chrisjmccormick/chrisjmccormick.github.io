---
layout: post
title:  "Refactoring the Attention Equations: Patterns and Messages"
date:   2025-02-19 23:00:00 -0800
comments: true
image:
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

_I figure I'll publish another post on this each day until it's fully birthed. Up ahead--practical applications, illustrations, experimental results, and who knows what else!_

The original formulation of multi-head attention from Vaswani et. al concatenates multiple attention heads before applying the output projection matrix $W^O$:

$$
\begin{aligned}
    \text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
    \text{head}_i &= \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) \\
    \text{Attention}(Q, K, V) &= \text{softmax} \left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\end{aligned}
$$


## 1. Splitting $W^O$ by Head

By splitting $W^O$ by head, we can replace the concatenation with a sum: 

$$
\text{MultiHead}(X) = \sum_{i=1}^{h} \text{head}_i W^O_i
$$

Instead of concatenating heads, we project each head separately using $W^O_i$. 

MultiHead(Q, K, V) is a way of indicating that these three attention inputs can have different sources, such as in multimodal attention or cross-attention. But I'll simply write $X$ as the input for simplicity. 

## 2. Separating the Value Projection from the Scores

We replace the "Attention" function with just the "scores", and move out the Value projection.

$$
\begin{aligned}
    \text{MultiHead}(X) &= \sum_{i=1}^{h} \text{scores}_i \cdot \text{values}_i \\\\
    \text{scores}_i &= \text{softmax} \left(\frac{X W_i^Q (X W_i^K)^T}{\sqrt{d_k}}\right) \\\\
    \text{values}_i &= X W_i^V W^O_i
\end{aligned}
$$

## 3. Introducing Messages $M$

So far, we have treated $W^V$ and $W^O$ as separate transformations. However, they can be seen as a **low-rank decomposition** of a larger transformation. Defining a single matrix $W^M$, we rewrite:

$$
M_i = X W^M_i
$$

Each row in $M_i$ represents a **message** passed between tokens in the attention mechanism. Attention scores determine **how much influence each message has** on a given token.

$$
\begin{aligned}
    \text{MultiHead}(X) &= \sum_{i=1}^{h} \text{scores}_i \cdot \text{messages}_i, \\\\
    \text{scores}_i &= \text{softmax} \left(\frac{X W_i^Q (X W_i^K)^T}{\sqrt{d_k}}\right) \\\\
    \text{messages}_i &= X W^M_i
\end{aligned}
$$


## 4. Introducing Patterns $P$

Instead of separately computing queries and keys, we can merge their weight matrices into a single matrix $W^P_i$:

$$
W^P_i = W^Q_i (W^K_i)^T
$$

To compute attention scores we first compute the attention head patterns:

$$
P_i = X W^P_i
$$



Now, attention logits can be rewritten as:

$$
\text{logits}_i = P_i X^T
$$

Each row of $P_i$ represents a **pattern** that the attention head is searching for in the sequence.

This reformulation keeps attention in **model space**, making it more interpretable.


## 5. Updated Multi-Head Attention Formulation

Here is the final version, in rough English. Note that this is for full self-attention, meaning all queries against all keys. 

$$
\begin{aligned}
    \text{MultiHead}(X) &= \sum_{i=1}^{h} \bigl(\text{scores}_i \cdot \text{messages}_i \bigr) \\\\
    \text{scores}_i &= \text{softmax} \left(\frac{\text{patterns}_i \cdot X^T}{\sqrt{d_k}} \right) \\\\
    \text{patterns}_i &= X W^P_i \\
    \text{messages}_i &= X W^M_i
\end{aligned}
$$

<br/>

---


Or in variables as:

$$
\begin{aligned}
    \text{MultiHead}(X) &= \sum_{i=1}^{h} \alpha_i M_i \\\\
    \alpha_i &= \text{softmax} \left(\frac{P_i X^T}{\sqrt{d_k}} \right) \\\\
    P_i &= X W^P_i \\
    M_i &= X W^M_i
\end{aligned}
$$

Where:

$$
\begin{aligned}
    \text{Tokens}& \quad\quad X \in \mathbb{R}^{T \times d_{\text{model}}} \\
    \text{Scores}& \quad\quad \alpha_i \in \mathbb{R}^{T \times T} \\
    \text{Patterns}& \quad\quad P_i \in \mathbb{R}^{T \times d_{\text{model}}} \\
    \text{Messages}& \quad\quad M_i \in \mathbb{R}^{T \times d_{\text{model}}} \\
\end{aligned}
$$

<br/>

For the single query version, replace $P_i$ with $p_i = x_qW^P_i$
