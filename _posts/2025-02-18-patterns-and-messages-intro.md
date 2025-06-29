---
layout: post
title:  "Patterns and Messages: A New Framing of Transformer Attention"
date:   2025-02-18 12:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1o954cnOGwZyxq7JW6mgLqXaNjrfw_cRM
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

I recently had a series of "aha!" moments around the Attention equations that's lead to some exciting weeks of research and insight.

The core revelation is that the way we've been taught to think about Transformers reflects an emphasis on computational efficiency and GPU optimization, and that if we step back from the implementation details we can arrive at a cleaner, more intuitive representation.

We'll reformulate the equations in a way that is less efficient, but mathematically equivalent, and which (I strongly believe) provides a better mental model for understanding Transformers, and Attention in particular.



_by Chris McCormick_

## Contents

* TOC
{:toc}


## Insights from Mechanistic Interpretability


I couldn't believe I hadn't heard these ideas before, and initially was left thinking that I might have a big discovery on my hands!

I eventually found that these ideas are well understood and leveraged by the field of Transformers research called "Mechanistic Interpretability", which aims to fully reverse engineer and explain the 'mechanisms' of Transformer models (as opposed to standard interpretability, which has more to do with explaining to end users how a model made its decision).

**Transformer Circuits**

Particularly, the paper "[Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" from Anthropic builds a mathematical framework for understanding Attention and Transformers using the same insights (and takes them much further, of course!).

That may have been bad for my ego, but really it's a huge win--the perspectives here aren't some wild idea of mine, they are a fundamental part of the tools used by respected researchers to reflect on how Transformers work!

**Interpretability vs. Accessibility**

A number of the most satisfying insights from interpretability research have made it into popular tutorials on Transformers, such as how some attention heads correspond nicely to syntactic roles, and the idea of the Feed Forward Networks being where the "memory" of the Transformer lies.

I think we have been missing out on some additional big insights that can be derived from the ways this field formulates the equations and illustrates the architectures.

In this series of posts, I'll share my own journey towards these ideas and how I think they can make Transformers more teachable and hopefully provide a more intuitive and insightful mental model for all of us to use.

It's a small shift from how interpretability research uses them to deeply interrogate the mechanisms of Transformers, and I'm hoping it will lead to more widespread adoption to the perspective already in use by the researchers who best understand the Transformer's "inner workings".

**Overview**

The remainder of this post is an outline of the posts and their key insights. Note that I've made the illustrations deliberately small--they're intended as a peek of what's ahead rather than explanations to be studied.

For each post, I've also included references to the corresponding points in the "Key Takeaways" section of the Transformer Circuits paper.


## Part 1 - Breaking Apart the Heads

([Link to post](https://mccormickml.com/2025/02/18/patterns-and-messages-part-1-wo-i/))

It all begins with recognizing that:

**Attention has a per-head output matrix, $ W^O_i$**

<br/>


<img src="https://lh3.googleusercontent.com/d/1yL8t8FBrAOw1G1060ar2KZ6FU1whPxMP" alt="Per head output matrices" width="400"/>



I think that if we adjust the equations to clarify this detail, that alone can significantly simplify and improve how we teach and think about MultiHead Attention.

This change makes it clear that:

**The multiple heads in Attention operate independently and can be cleanly separated from each other.**



<img src='https://lh3.googleusercontent.com/d/1AfS0ML6LTFPWhsDNSkLtcifYiTpFGggD' alt='Single head attention' width='170'/>



The outputs are simply summed together (I usually think of $W^O$ as re-projecting and recombining the values, and "recombining" seems to imply a more complex process than just summing).

This simple explanation is obscured by the way we concatenate all of the weight matrices together in practice, and especially by how we concatenate the value vectors prior to the output matrix.



Once separated, it also becomes more clear that:

**Heads consist of two independent processes: Query-Key and Value-Output**



<img src='https://lh3.googleusercontent.com/d/1AeHb7ZhG1pvGsVJHNGj3VDH-xb19up4f' alt='Separating the Query-Key and Value-Output processes' width='170'/>



I think the novelty there is that the output matrix really belongs to the Value vectors, and that it is conceptually clearer to apply the attention scores _after_ the Output step rather than before (as we do in practice).

From the Circuits paper:

> * Attention heads can be understood as independent operations, each outputting a result which is added into the residual stream. Attention heads are often described in an alternate “concatenate and multiply” formulation for computational efficiency, but this is mathematically equivalent.
>
> * ...
>
> * Attention heads can be understood as having two largely independent computations: a QK (“query-key”) circuit which computes the attention pattern, and an OV (“output-value”) circuit which computes how each token affects the output if attended to.

(I'll explain what they mean by the "residual stream", I think it's a very useful concept).

These adjustments don't deviate much from our current framing of Attention, and I think they would be beneficial to include in our existing tutorials on the topic.

This reformulation does motivate some further shifts, though, which can offer a fresh perspective for those who find it valuable.

## Part 2 - Patterns and Messages

([Link to post](https://mccormickml.com/2025/02/18/patterns-and-messages-part-2-token-communication/))

The separation of the Query-Key and Value-Output processes makes it clear that $W^V$ and $W^O$ are in fact a **low-rank decomposition** of some larger matrix.

Less obvious in the math, but still a logical next step, is to recognize that $W^Q$ and $W^K$ are a low rank decomposition as well.

These are referred to by the interpretability community quite sensibly as the $W^{QK}$ and $W^{OV}$ matrices.

In this part of the series, I'll:

* Show how we arrive at these matrices, and what they represent.
* Propose how we might improve our metaphors for Attention by giving unique names to these "conceptual" matrices.


**Messages**

The merged $W^{VO}_i$ is used to produce the per-head, per-token, model space vector which contains the **message** that a token will communicate if selected.


<img src='https://lh3.googleusercontent.com/d/1o954cnOGwZyxq7JW6mgLqXaNjrfw_cRM' alt='Merging the value and ouptut projections into the message projection' width='400'/>


"Message" borrows from Graph Neural Networks, and fits the communication terminology used throughout Transformer Circuits.


**Patterns**

The merged $W^{QK}_i$ matrix is used to produce the **pattern** vector used for selecting and weighting the messages.


<img src='https://lh3.googleusercontent.com/d/18fiJf-r1QIrSkHcbyaiNCYNLceKIdqN2' alt='Merging the query and key projection matrices into the pattern projection' width='400'/>




We name other transitory vectors in a similarly straightforward manner, e.g., "the activations" or "the scores". $p$ is simply "the pattern".


From _Transformer Circuits_:

> * Key, query, and value vectors can be thought of as intermediate results in the computation of the low-rank matrices $W_Q^TW_K$​ and $W_OW_V$​. It can be useful to describe transformers without reference to [the key, query, and value vectors].

There are additional benefits which I felt deserved their own posts, and I've summarized them in the following sections.

First, I think this framing gives us a more "fundamental" version of Attention which can serve as a starting point for considering various ways to make it more efficient.


## Part 3 - Alternative Decompositions

([Link to post](https://mccormickml.com/2025/02/19/patterns-and-messages-part-3-alternative-decompositions/))

The merged-matrix concept is a helpful mental model, but not a good implementation. We know we have (at least) three issues to resolve:

1. **Compute Cost** - $W^P_i$ and $W^M_i$ are expensive to project on to,
2. **Cache Size** - We want to cache the calculated patterns and messages, but they're very large, and this becomes a substantial problem with longer sequence lengths.
3. **Low Rank** - We also believe that $W^P_i$ and $W^M_i$ are over-parameterized--that the heads should operate on far fewer features than the full embedding space.

Our standard query-key, value-output decomposition is one approach for addressing all three, but there are alternatives!

In this part of the series we'll look at some existing approaches, but from the Patterns and Messages perspective. Specifically, Group Query Attention (GQA), and MultiHead Latent Attention (MLA). 

Personally, I found Patterns and Messages to be a remarkably satisfying way of motivating and understanding MLA! (Which makes sense given that MLA is actually what inspired me to think in this direction of merging the matrices)


## Part 4 - Attention as a Dynamic Neural Network

([Link to post](https://mccormickml.com/2025/02/19/patterns-and-messages-part-4-a-dynamic-neural-network/))

Reducing the number of Attention matrices from four to two makes it easier to view an Attention head as a standard neural network, except whose weights are dynamically created by the projection matrices during inference.

It draws a natural parrallel between Attention and the FFN:
* The FFN is a "statically learned" neural network whose weights are learned during training.
* An attention head is a "dynamically learned" neural network whose weights are built up during inference (the pattern vectors are the input neurons and the messages are the output neurons).

See the post for details, but here's a quick visualization that might help the concept land.

The FFN:

<img src='https://lh3.googleusercontent.com/d/1ZZCuWzc_Hiz75SZXm0k77RgN2yMHyip1' alt='An FFN with SwiGLU' width='400'/>

An attention head:

<img src='https://lh3.googleusercontent.com/d/1r5HV_P93C_oD66grsX3IpPCrAre0ENNd' alt='Evaluating the updated Attention Head Neural Network' width='400'/>

Note that the attention network is not formed by the learned _projection matrices_, but by the projected tokens at runtime.

It's a fresh perspective and motivates some interesting questions--for example, do we really need to add every token to every head, or can we drop those that aren't applicable?

## Part 5 - The Residual Stream

([Link to post](https://mccormickml.com/2025/02/20/patterns-and-messages-part-5-the-residual-stream/))

Something I find really helpful about this merged-matrix perspective is that it puts everything in "model space". The patterns and messages and their projection matrices all have the same length as the word embeddings.

Once you view attention this way, it becomes clear that the entire transformer process is additive. The output word vector is nothing more than the **input embedding** plus a weighted sum of **all messages** plus a weighted sum of **all output neurons**.

<img src='https://lh3.googleusercontent.com/d/1_duz3CEV6DUV27NB31IeeKZUKLX8XSZ_' alt='The residual stream drawn as a stack of the neural memories, messages, and input vector, with scores and activations, and example dimensions' width='300'/>

This concept of each component adding to a growing stack of information is referred to as the "residual stream", highlighting how it serves as the communication channel between different parts of the model.

I think that "residual" can be a misleading metaphor, and the way Transformers are drawn in Mechanistic Interpretability research helps resolve this.

Instead of a line drawn through the center of the components, we can highlight their additive nature by putting them off to the side, each one reading from and writing back to the Stream.

<img src='https://lh3.googleusercontent.com/d/12UkXHJJx90m0dtPYEDLcThjkuD_yWWSy' alt='An Attention Head Network reading and writing to the stream' width='300'/>




From the Transformers Circuits paper:

> * All components of a transformer (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model.

## Part 6 - Vocabulary-Based Visualizations

([Link to post](https://mccormickml.com/2025/02/21/patterns-and-messages-part-6-vocabulary-based-analysis/))

Because the pattern and message vectors live in model space—the same space as the vocabulary embeddings—we can sometimes compare them directly to actual words.

This only works with models that tie their input and output embeddings (like GPT-2, T5, and the recent Microsoft Phi-4-mini), but when it works, it's satisfying to see! You can visualize what a head is “looking for” or “saying” in terms of top-matching words from the vocabulary. I've found some heads that clearly align with specific topics (e.g., geography, religion), and created illustrations of how meaning evolves across decoder layers.

It's not a catch-all interpretability tool—it only reveals semantic behavior, and there's plenty of non-semantic work being done—but where it applies, it offers some nice intuition.

## What's Next?

Throughout the series, I've weaved in some different ideas / speculations I've had that were inpsired by this alternative perspective. I hope I get the chance to discuss and explore them further!

## Conclusion

All of this shift in perspective was incredibly exciting to me--it felt earth shattering! I'm eager to see how others feel about it.

Perhaps this is simply me discovering that I'm an interpretability nerd at heart, and these posts will serve mostly as an introduction for others who, like me, were previously unfamiliar with that field of research.

I suspect it's more than that, though, and that we've all been missing out on these insights because they've only been presented as tools for going "down the rabbit hole" of interpretability.

(The series covers all of the "key takeaways" from the Transformer Circuits paper except for two--the ones which go deeper into interpretability and discuss compositing attention heads and constructing chains of matrices)

I think it's valuable for all of us to understand that a token's message influences other tokens, which in turn send messages to other tokens, and so on, allowing for complex behaviors.

The "Transformer Circuits" paper provides a framework for tracing those messages and diving deeper for those who want to, but I think that everyone could benefit from just wading in to the shallow end.

> Side Note: I owe Nick a beer. On multiple occassions he tried to get me to read the work of Chris Olah (a popular blogger, co-founder of Anthropic, and credited as the founder of Mechanistic Interpretability). "Seems like if we're trying to explain this stuff we ought to study the research of people trying to understand it." I consider it quite the achievement that I was able to ignore such logical advice.

**Cite**  

McCormick, C. (2025). *Patterns and Messages: A New Framing of Transformer Attention.*  
Retrieved from https://mccormickml.com/2025/02/18/patterns-and-messages-intro/

