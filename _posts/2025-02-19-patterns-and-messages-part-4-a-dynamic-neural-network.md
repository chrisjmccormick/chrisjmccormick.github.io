---
layout: post
title:  "Patterns and Messages - Part 4 - Attention as a Dynamic Neural Network"
date:   2025-02-19 23:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1r5HV_P93C_oD66grsX3IpPCrAre0ENNd
tags: Transformers, Attention Mechanism, Neural Network Architecture, Dynamic Networks, Interpretability, Residual Stream, Token Efficiency, Multi-Head Attention, Mixture of Experts, GQA
---

When you reduce Attention down to two matrices instead of four, the pattern and message vectors represent a more familiar architecture--they form a neural network, whose neurons are created dynamically at inference time from the tokens.

This draws a nice parallel between the Feed Forward Neural Network and this "Attention Head Neural Network".

The FFN is a large "static" neural network whose input and output weights are learned during training and then fixed. It uses the SwiGLU activation function.

The AHN is a dynamically built neural network whose input and output weights are created during inference by projecting the tokens. It uses the SoftMax activation function.

For a given head, the cached key vectors are the input neurons, and the cached value vectors are the output neurons.

It's a little cleaner, though, when you use the pattern and message vectors, as I'll show here.

_by Chris McCormick_

## Contents

* TOC
{:toc}

## Feed Forward Neural Network

First, just to highlight the similarity, here is how I might illustrate the FFN in Llama 3 8B, which has an embedding size of `4,096` and an FFN size (inner dimension) of `14,336`.

I've included the dimensions because I find them helpful for understanding "what I'm looking at" in an illustration. Also, note that I've chosen to orient the "matrices" based on what's convenient for the illustration rather than what's required for correct matrix multiplication.

To evaluate the FFN:

The current input token is multiplied by the input neurons and their gates, resulting in an activation value for each neuron.

These activations are multiplied against their respective output neurons, which are summed to produce the output.

<img src='https://lh3.googleusercontent.com/d/1ZZCuWzc_Hiz75SZXm0k77RgN2yMHyip1' alt='An FFN with SwiGLU' width='700'/>

In the Attention network, the input weights are the pattern vectors and the output weights are the messages. I'll go into more detail in the next section, but wanted to include this here for easy comparison.

<img src='https://lh3.googleusercontent.com/d/1r5HV_P93C_oD66grsX3IpPCrAre0ENNd' alt='Evaluating the updated Attention Head Neural Network' width='700'/>

I drew a box around the activations here to highlight that, unlike the FFN, they are normalized.

The attention neural network starts out small, but steadily grows in size as  we process the sequence.

Also note that there are multiple of these per layer, corresponding to the number of heads, but only one FFN per layer.

## Attention Head Neural Network




Within an attention head, when processing an input token, attention can be thought of in terms of two steps:

1. **Append** a **new input** and **output** neuron to the network by projecting the input vector onto the Pattern and Messages spaces, respectively.
2. Evaluate the network on the input vector.

I've illustrated these two steps below. The text sequence is "Llamas do not output words" (unless they have version numbers! 😉), and the current input token is "words".

The subscript $i$ refers to the head number, and I include it everywhere to reinforce that we are looking at a single head.



**Step 1: Grow the Network**

First, grow the network by projecting the input token to get its pattern vector and message vector, and append these to their respective matrices.



<img src='https://lh3.googleusercontent.com/d/1y2Jyu6E9O0sIqWGEN72PyB5CokEOJJYK' alt='Adding a token to the Attention Head Neural Netork' width='400'/>

**Step 2: Evaluate**

Second, evaluate the updated network on that same input vector.

<img src='https://lh3.googleusercontent.com/d/1r5HV_P93C_oD66grsX3IpPCrAre0ENNd' alt='Evaluating the updated Attention Head Neural Network' width='800'/>

The output vector for this head and the others are all added to the current "residual stream". The residual stream is simply the input embedding with all of the head outputs and FFN outputs added on top. I'll explore this more in the next post.

## Pros and Cons

The difference between this and our existing framing seems subtle--you could replace "patterns" and "messages" with "keys" and "values" and achieve similar insights. Incorporating $W^Q$ and $W^O$ into that picture unnecessarily complicates things, though.

Mostly, it seems to provide a fresh perspective that can inspire creativity.



### Benefits



Here are some thoughts I've had around it.

**Group Query Attention as Weight Tying**

This neural network formulation helped me better understand GQA. In Llama 3 8B, groups of four AHNs have distinct pattern vectors / input neurons, but they all share the same messages / output neurons. It allows for four different blends of the same messages.

**Token Filtering**

As we get further along in processing / generating text, these AHNs get very big and expensive to compute! Do we really need to add every token to all of them? Can we recognize when $W^M_i$ isn't relevant to a token, or when the resulting message $m_i$ is meaningless, and refrain from including it?

**Gating**

We're dynamically creating input and output neurons; should we be dynamically creating gates for them as well?

**Input Routing**

Seeing the heads as neural networks draws the comparison to Mixture of Experts. MoE works by clustering related neurons so that we can filter inputs by comparing to their centroid.
Are the pattern vectors for a head similar enough to each other to have a representative centroid we could use for routing the queries?





### Concerns

The main problem with this perspective, which applies to patterns and messages in general, is that it hides the fact that they are low rank.

It's important to remember that, unlike the FFN, these Attention Head Networks have been "bottlenecked" through low rank decomposition so that they only modify a particular "subpace" of the token vector.

In the same way that LoRA limits how much impact the fine-tuning process has on the model weights, these AHNs have limited impact on the token compared to the FFN.

## Conclusion

Collapsing down from "QKVO" to "PM" helps us see attention from a new angle, as dynamically building a neural network from each new token.

This framing encourages us to ask questions like:

* Does every context token need to add a message to every head?
* Does every input token need to be routed to every head?
* And / or, should we gate the scores instead of normalizing them?

In the next post, we'll look at another concept from Mechanistic Interpretability called "The Residual Stream", which pulls the concepts in this post together to describe the behavior of a layer and the Transformer overall.

## Appendix - Notation

For those who'd prefer a mathematical definition...

In a Decoder text generation model, the current input vector is used in all three roles, $x_q = x_k = x_v $, but I'll keep them separate just to help us map things back to what we're used to.


**Step 1: Grow the Network**

The pattern vector $p_i$ is produced by projecting the key token onto the Pattern space:

$p_i = x_kW^P_i$

The message vector $m_i$ is produced by projecting the value token onto the Message space:

$m_i = x_vW^M_i$

These get appended to the vectors in the cache (the left arrow refers to "updating" / replacing the matrix):

$P_i \gets \text{Concat}(P_i, \; p_i)$

$M_i \gets \text{Concat}(M_i, \; m_i)$




**Step 2: Evaluate the Network**

Calculate the activations by multiplying the query token against all of the pattern vectors, and then normalizing those activations with the SoftMax.

$\alpha_i = \text{SoftMax} \left(\frac{x_q P_i^{'}}{\sqrt{d^{h}}} \right)$

And then the output of the network (i.e., the output of the Attention head) is:

$o_i = \alpha_iM_i$
