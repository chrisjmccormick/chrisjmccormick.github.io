---
layout: post
title:  "Rethinking Value and Output Matrices in Multi-Head Attention: A Low-Rank Decomposition Perspective"
date:   2025-02-17 17:00:00 -0800
comments: true
image: 
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

_After noodling on this all day, I finally decided to let GPT turn it into a blog post for me so I could just publish it and let it go for the night! I'll come back to refine it._ 

## The Hidden Structure in $W_O$

When we write out Transformer attention, we usually express the final output projection as a single learned matrix, $W_O$. But if you take a closer look, you’ll notice that $W_O$ isn’t a single transformation—it’s actually the **concatenation of independent per-head projections**.

Standard attention maps input sequences to output embeddings like this:

1. **Project input $X$ into value space** via $W^V_i$, for each head $i$:  
   $$ V_i = X W^V_i $$
2. **Apply attention scores $a_i$ to weight value vectors**:
   $$ O_i = a_i V_i W^O_i $$
3. **Concatenate all heads and apply $W_O$**:
   $$ O = [O_1, O_2, ..., O_h] W_O $$

But here’s the key realization: **each head has its own $W^O_i$, meaning $W_O$ is just the concatenation of per-head matrices**:
   $$ W_O = [W^O_1, W^O_2, ..., W^O_h] $$

That means we can stop thinking of $W_O$ as a monolithic matrix and instead focus on its individual per-head components.

## A New Perspective: The Emergence of $W_M$

If we take this further, we can see that the value projection $W^V_i$ and output projection $W^O_i$ for each head can be **collapsed into a single transformation**:

$$ W^M_i = W^V_i W^O_i $$

This means that rather than thinking in terms of separate “value” and “output” projections, we can think of each head as applying a **low-rank transformation** to the input, where:

- $W^V_i$ **compresses** the input ($d_{model} \to d_v$)
- $W^O_i$ **re-expands** it ($d_v \to d_{model}$)
- The product, $W^M_i$, **directly maps $d_{model} \to d_{model}$, but with rank at most $d_v$**

This formulation is important because it makes explicit something that was already happening implicitly: each head’s transformation is **rank-limited** by $d_v$, even though it operates in model space.

## Reformulating Multi-Head Attention (Without Concatenation!)

Using this decomposition, we can rewrite multi-head attention **without concatenation**:

1. Compute **modifier vectors** per head:
   $$ M_i = X W^M_i $$
2. Apply attention scores to get the **final modification**:
   $$ m^f_i = a_i M_i $$
3. Sum all heads to get the final output:
   $$ O = \sum_{i=1}^{h} m^f_i $$

Or in code:

```python
# Instead of concatenating across heads:
M_i = X @ W_V[i] @ W_O[i]  # No need to stack or reshape
m_f_i = attention_scores[i] @ M_i  # Directly operates in model space
output = sum(m_f_i)  # Final sum replaces concatenation
```

This is mathematically equivalent to standard attention but removes the need for concatenation—it directly models each head’s contribution in full model space.

## Why This Matters

This view has several implications:

- **No need for a final projection step:** Each head already contributes directly to $d_{model}$, so there’s no need for concatenation.
- **Explicit rank constraints:** Each head is inherently limited to rank $d_v$, which could inform future modifications like dynamic rank selection or alternative factorization techniques.
- **A different way to think about head diversity:** Instead of each head contributing an independent vector slice, each head contributes an independent rank-limited transformation.

### Final Thought

This reframing doesn’t change how Transformers are implemented, but it changes how we think about them. **Instead of seeing multi-head attention as a concatenation process, we can see it as a sum of independent low-rank modifications.**

Maybe the next step is asking: what happens if we structure or regularize $W^M_i$ differently? Would enforcing sparsity or alternative decompositions lead to better architectures? There’s a lot to explore here.
