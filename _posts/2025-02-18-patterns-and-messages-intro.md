---
layout: post
title:  "Patterns and Messages - Intro and Overview"
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

The remainder of this post is an outline of the posts and their key insights.

I've also included references to the corresponding points in the "Key Takeaways" section of the Transformer Circuits paper.


## Part 1 - Breaking Apart the Heads


It all begins with recognizing that:

**Attention has a per-head output matrix, $ W^O_i$**

<br/>


<img src="https://lh3.googleusercontent.com/d/1yL8t8FBrAOw1G1060ar2KZ6FU1whPxMP" alt="Per head output matrices" width="400"/>



I think that if we adjust the equations to clarify this detail, that alone can significantly simplify and improve _all of our existing tutorials_ on MultiHead Attention.

This change makes it clear that:

**The multiple heads in Attention operate independently and can be cleanly separated from each other.**



<img src='https://lh3.googleusercontent.com/d/1AfS0ML6LTFPWhsDNSkLtcifYiTpFGggD' alt='Single head attention' width='170'/>



The outputs are simply summed together (I usually think of $W^O$ as re-projecting and recombining the values, and recombining" seems to imply a more complex process than just summing).

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

The separation of the Query-Key and Value-Output processes makes it clear that $W^V$ and $W^O$ are in fact a **low-rank decomposition** of some larger matrix.

Less obvious in the math, but still a logical next step, is to recognize that $W^Q$ and $W^K$ are a low rank decomposition as well.

These are referred to by the interpretability community quite sensibly as the $W^{QK}$ and $W^{OV}$ matrices.

In this part of the series, I'll:

* Show how we arrive at these matrices, and what they represent.
* Propose how we might improve our metaphors for Attention by giving unique names to these "conceptual" matrices.
* Highlight some of the practical benefits to viewing Attention from this perspective.  


**Messages**

The merged $W^{VO}_i$ is used to produce the per-head, per-token, model space vector which contains the **message** that a token will communicate if selected.


<img src='https://lh3.googleusercontent.com/d/1o954cnOGwZyxq7JW6mgLqXaNjrfw_cRM' alt='Merging the value and ouptut projections into the message projection' width='400'/>


"Message" borrows from Graph Neural Networks, and fits the communication terminology used throughout Transformer Circuits.


**Patterns**

The merged $W^{QK}_i$ matrix is used to produce the **pattern** vector used for selecting and weighting the messages.


<img src='https://lh3.googleusercontent.com/d/18fiJf-r1QIrSkHcbyaiNCYNLceKIdqN2' alt='Merging the query and key projection matrices into the pattern projection' width='400'/>




We name other transitory vectors in a similarly straightforward manner, e.g., "the activations" or "the scores". $p$ is simply "the pattern".


**Decomposition Strategies**

Our standard QK, VO matrix decomposition is just a specific design choice, and there may be other solutions.

It can be helpful to view and understand current variants of MultiHead Attention from this perspective--particularly MultiHead Latent Attention (MLA) which inspired me to think in this direction.

From _Transformer Circuits_:

> * Key, query, and value vectors can be thought of as intermediate results in the computation of the low-rank matrices $W_Q^TW_K$​ and $W_OW_V$​. It can be useful to describe transformers without reference to [the key, query, and value vectors].

There are additional benefits which I felt deserved their own posts, and I've summarized them in the following sections.



## Part 3 - Attention as a Dynamic Neural Network

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

## Part 4 - The Residual Stream

Something I find really helpful about this merged-matrix perspective is that it puts everything in "model space". The patterns and messages and their projection matrices all have the same length as the word embeddings.

Once you view attention this way, it becomes clear that the entire transformer process is additive. The output word vector is nothing more than the **input embedding** plus a weighted sum of **all messages** plus a weighted sum of **all output neurons**.

> Side Note: The layer normalizations interfere with this interpretation, but not enough to invalidate it, it would seem.

The input vector plus its growing pile of information is referred to as the "The Residual Stream".


From the Transformers Circuits paper:

> * All components of a transformer (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model.

(Note the use of "communicate", "reading and writing", and "communication channels")


**Confusion over "Residual"**

This framing corrects what I've found to be some rather confusing terminology and conventions:
* The term "residual" is used to describe what's left after something else is removed, like a "residue" left behind by something.
* "Residual connection", and the line we draw for this, suggests to me that some sort of residue travels from the input vector to be applied to the output.


<img src='https://lh3.googleusercontent.com/d/1pjg9M9s5FouVd6YZXiTWE_ASLvVZl10-' alt='The standard way to illustrate a deep neural network and the residual connection' width='200'/>



Neither the metaphor nor the diagram seem consistent to me with the implementation:

In deep neural networks, vectors do not flow through them, as they did with classic MLPs. Instead, each component's job is to produce an _adjustment_ to be applied (added on) to the input vector.

The "stream" concept makes this clear, and is illustrated by drawing a straight line from input to output, with each of the components reading from the stream and then adding something back on to it.

This illustration from the original framework paper, [here](https://transformer-circuits.pub/2021/framework/index.html), shows the token embedding at the start of the stream, and then the multiple attention heads reading it and producing something to add to it.



<img src='https://lh3.googleusercontent.com/d/1AmTyP8DnG2FRr5L83bTX6rJOOYy1jY1o' alt='Residual stream illustration from Transformer Circuits' width='800'/>



The FFN would be the next item on the line, as a single block (vs. the multiple heads).

> Side Note: Does the term "residual", implying "residue", make anyone else queazy, or is that just me?
>
> I'd prefer something like "input-output stream", or at the very least "Res" Stream, like ResNet.

## Part 5 - Vocabulary-Based Visualizations

Because the pattern and message vectors live in model space—the same space as the vocabulary embeddings—we can sometimes compare them directly to actual words.

This only works with models that tie their input and output embeddings (like GPT-2, T5, and the recent Microsoft Phi-4-mini), but when it works, it's incredibly satisfying. You can visualize what a head is “looking for” or “saying” in terms of top-matching words from the vocabulary. I've found some heads that clearly align with specific topics (e.g., geography, religion), and created illustrations of how meaning evolves across decoder layers.

It's not a catch-all interpretability tool—it only reveals semantic behavior, and there's plenty of non-semantic work being done—but where it applies, it offers some nice intuition.

## Part 6 - Further Research

To minimize tangents into many speculative insights I've taken from all of this, I'm trying to move those into their own post where they can be shared and discussed with (and refuted by) anyone actually interested. 😊

Briefly:

**Empty Messages**

Trying to compare the messages to the vocabulary, I found that many messages strongly resembled either the most frequent or the most infrequent tokens in the vocabulary. I'm wondering if these messages are meaningless "no-ops", where the token has decided it has nothing to contribute via that head. If that's true and we could detect these, we could avoid adding them to the KV cache.

**Routing Attention**

If it's true some tokens want to exclude themselves from some heads, or that sometimes an entire head needs to turn itself off, then the SoftMax activation feels like a bizarre choice. Both $W^P$ and $W^M$ have to learn to detect this state in order for the head to contribute (harmless noise?) to the stream, and we also give all heads equal weight.

A gated activation function (with the output divided by the number of tokens?) would separate out the "shut-off" role, allow messages to be added or subtracted, and provide more expression in the weighting of the messages.

I think the point of SoftMax / normalizing the attention scores is to encourage the heads to learn different behaviors? MoE solved this by adding a load balancing objective during training--maybe that could be used here, and even let us avoid adding tokens to heads where they aren't needed?

**Fuse and Rank Reduce**

SVD analysis shows that $W^M$ and $W^Q$ consistently have lower effective rank than their decomposed form (though usually only slightly). We can take a pre-trained model, fuse the matrices, then use SVD to decompose them again while dropping some of the values to get a smaller weight matrix. Not sure how meaningful the savings is, though, or whether 'effective rank' is a good indicator that we can safely drop those values.





## Conclusion

All of this shift in perspective was incredibly exciting to me--it felt earth shattering! I'm eager to see how others feel about it.

Perhaps this is simply me discovering that I'm an interpretability nerd at heart, and these posts will serve as mostly an introduction for others who, like me, were previously unfamiliar with that field of research.

I suspect it's more than that, though, and that we've all been missing out on these insights because they've only been presented as tools for going down the rabbit hole of interpretability.

I've referenced all of the "key takeaways" from the Transformer Circuits paper except for two, which discuss compositing attention heads and constructing chains of matrices.

I think it's valuable for all of us to understand that a token's message influences other tokens, which in turn send messages to other tokens, and so on, allowing for complex behaviors.

The "Transformer Circuits" paper provides a framework for tracing those messages and diving deeper for those who want to, but I think that everyone could benefit from just wading in to the shallow end.

> Side Note: I owe Nick a beer. On multiple occassions he tried to get me to read the work of Chris Olah (a popular blogger, co-founder of Anthropic, and major contributor to interpretability research). "Seems like if we're trying to explain this stuff we ought to study the research of people trying to understand it." I consider it quite the achievement that I was able to ignore such logical advice.


