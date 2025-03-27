---
layout: post
title:  "Patterns and Messages - Part 5 - The Residual Stream"
date:   2025-02-20 23:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1DIHw1vkkz49UC7nodMhqpjb4gKz38cdb
tags: Transformers, Attention Mechanism, Neural Network Architecture, Interpretability, Residual Stream, Multi-Head Attention
---

Something I find really helpful about this merged-matrix perspective is that it puts everything in "model space". The patterns and messages and their projection matrices all have the same length as the word embeddings.

Once you view attention this way, it becomes clear that the entire transformer process is additive. The output word vector is nothing more than the **input embedding** plus a weighted sum of **all messages** plus a weighted sum of **all output neurons**.


<img src='https://lh3.googleusercontent.com/d/1_duz3CEV6DUV27NB31IeeKZUKLX8XSZ_' alt='The residual stream drawn as a stack of the neural memories, messages, and input vector, with scores and activations, and example dimensions' width='500'/>



Some notes: on the illustration:

* I'm using the values calculated in the prior post for Llama 3 8B. A sequence of 8,192 tokens translates into 8M message vectors.
* I've labeled the output neurons as "neural memories" following the paper _Transformer Feed-Forward Layers Are Key-Value Memories_ ([arXiv](https://arxiv.org/abs/2012.14913 )).


_by Chris McCormick_



## Contents



* TOC
{:toc}



## The Residual Stream

This additive perspective tends to be hidden by our use of matrix multiplication, which on the outputs of networks combines the two operations of:

1. Applying scores / activation values to vectors
2. Summing the vectors together

(As we've seen, this is particularly true in Attention, where the ordering of the operations really buries this insight!)

The approach of instead keeping the contributing vectors **conceptually separate** is referred to in Mechanistic Interpretability as the "Residual Stream".


> Side Note: What about Layer Normalizations?
>
> This simple additive view isn't entirely correct, because in order to properly calculate the final output based on this collection of vectors, we'd need to insert the normalization steps in the appropriate places. 
>
> These normalizations are important to take into account when probing the stream, but otherwise don't seem to significantly break or invalidate this additive interpretation.



**Overview**

In this post, 

1. I'll share some difficulties I have with the "residual connection", and how the "stream" view addresses them.
1. Look at a different way of drawing the Transformer architecture to highlight the Stream.
2. Discuss implications for interpreting Transformers.

I misunderstood the residual connection for a long time (having written it off as "just something we do that makes training work better"), but eventually came to realize how critical it is for a correct mental model of the Transformer. Perhaps it's just me, but I think the "residual" terminology can be misleading. 




## Confusion over "Residual"

The "Stream" framing resolves what I've found to be some rather confusing terminology and conventions.

**Adding or Subtracting?**

First, in other contexts the word "residual" describes _what's left_ after something else is **removed**, like a "residue" left behind by something.

In contrast, in neural networks we feed in an input vector and the model produces a vector which we **add** back to the input vector.

This means the network has learned to produce an **adjustment** to make to the input, rather than directly produce a modified version of the input (as they did with classic MLPs).

I see the connection to the concept of a "residual"--it's conceptually as if we "removed the input from the output", and the residual is what's left.

In practice, we do the opposite.



**A Stream of Residue**

I hope you'll forgive my crudeness here, but does anyone else find the term unsettling? 

A deep neural network's job is to produce and apply "residue" to the input vector, layer by layer, until we have the output. 🤨 

We'll now further this metaphor by calling it a growing stream of residue along which the input travels. You're welcome.

Jokes aside, this innovation was _crucial_ for allowing us to train "deep" neural networks! And I think it deserves a more prominent role in our illustrations.

**Drawing Connections**

We also use the term "residual connection", reflecting how we illustrate it with a line drawn connecting the input and output of a component:


<img src='https://lh3.googleusercontent.com/d/1pjg9M9s5FouVd6YZXiTWE_ASLvVZl10-' alt='The standard way to illustrate a deep neural network and the residual connection' width='200'/>

It's really my fault for not having taken the time to learn the concept properly, but for a long time I inferred this line to mean that some kind of small "residue" is taken from the input and mixed back in to the output.

When I finally realized my mistake, it felt like a big revelation, because it completely changes the understanding of Transformer behavior.

The Attention and FFN modules don't radically transform the input vector, they gradually **refine** it through smaller adjustments.

It gives the mental image of something more like an **assembly line**, where the modules are each making their tweaks as the input vector travels down the line. We can re-draw the Transformer to reflect this.

## Alternative Illustration

The "stream" terminology captures this "assembly line" framing, and is illustrated by drawing a straight line from input to output, with each of the components reading from the stream and then adding something back on to it.

The below illustration comes from the original Transformer Circuits paper, [here](https://transformer-circuits.pub/2021/framework/index.html).

It shows the token embedding at the start of the stream, and then the multiple attention heads **reading from it**, each **producing** something which then gets **added** back to the stream.



<img src='https://lh3.googleusercontent.com/d/1AmTyP8DnG2FRr5L83bTX6rJOOYy1jY1o' alt='Residual stream illustration from Transformer Circuits' width='800'/>


Here is a version of some of my earlier illustrations which shows the two steps of an Attention Head Network interacting with the Stream.

From bottom up:

1. The lower block reads from the Stream to **project** and **append** a new pattern and message (as new neurons).
2. The upper block **evaluates** the network on the current state:
    1. **Matching** to the patterns,
    2. **Aggregating** the messages,
    3. **Writing** (adding) the result back to the Stream.

<img src='https://lh3.googleusercontent.com/d/12UkXHJJx90m0dtPYEDLcThjkuD_yWWSy' alt='An Attention Head Network reading and writing to the stream' width='400'/>

An important detail to remember here is that although the Messages are the same dimension as the input vector, they are **low-rank**, and only modify "some aspects" of the vector (they "write to a sub-space of the residual stream").

The next component on the stream is the FFN, shown here along with two attention heads.

<img src='https://lh3.googleusercontent.com/d/1DIHw1vkkz49UC7nodMhqpjb4gKz38cdb' alt='Attention and FFN on the Residual Stream' width='600'/>

The FFN output is, presumably, closer to full-rank. You can see this reflected by comparing the residual stream to the vocabulary--at various points, an FFN's output will change the "meaning" of the word vector from one word to another.

**Layer Normalization in the Stream**

Finally, in order to make this picture fully complete, we'd need to include the normalization blocks, which do actually "interrupt" the stream and would be drawn on top of it. 

Unlike the additive components, the norms do in fact take in the vector, perform a non-linear transformation, and output a "new" vector.

However, the transformation seems to be minor, and is a way for the model to keep things stable rather than change the vector in a meaningful way. 

My interpretation of their behavior is that they ensure that no individual dimension of the vector (i.e., no position within the length 4,096 vector) blows up in magnitude. It keeps all of the dimensions to a more consistent scale.

Dot products are highly sensitive to magnitude. Think back to logistic regression, and picture a housing price predictor that takes in square footage, number of bedrooms, and number of bathrooms. If we don't normalize those features, the number of beds and baths will have pretty much zero impact. (unless you're Little John in his 1 square-meter apartment in New York!)


## Interpretability

I'll dip into some insights from interpretability here (plus a little speculation), hopefully without taking things too far!


**Number of Vectors**

In the previous post, we worked out the size of the networks in Llama 3 8B. Those same numbers tell us the number of separate vectors hiding in this data stream when we've reached the output. 

For an input that's 8k tokens long, we have:

* 494K "neural memories" (each with $\text{rank} \leq \text{embedding size}$)
* 8M messages (each with $\text{rank} \leq \text{head size}$)



<img src='https://lh3.googleusercontent.com/d/1_duz3CEV6DUV27NB31IeeKZUKLX8XSZ_' alt='The residual stream drawn as a stack of the neural memories, messages, and input vector, with scores and activations, and example dimensions' width='400'/>


Of course, many of the scores and activations can / will be near enough to zero that we could ignore the contribution of their vectors.



**Compositions**

I think my illustration seems to imply that each message and each memory is individually meaningful. I'm not sure to what degree that's true, and suspect that it's largely incorrect.

In particular, interpretability research has found that the individual neurons in the FFNs are hard to interpret, I think implying that it's only the weighted combination of outputs which is meaningful. 

This may be true of the attention heads as well--that a head outputs something meaningful by assembling a composite of the messages, not necessarily selecting individual ones.

This suggests that the more meaningful vectors might actually be the "activation patterns".

**Activation Patterns**

One interesting insight from the Stream view is that, when processing a given token, almost all of the vectors in the data stream can be viewed as "fixed" / **constants**. 

<img src='https://lh3.googleusercontent.com/d/1wU_rH3Xba6ER7gf0ejjaTX2h7-YXU9DH' alt='The output of a Transformer broken down into the messages and neural memories and their scores, highlighting which are constant versus unique for each token' width='450'/>

The FFN output neurons are all fixed during training, and the token messages are fixed once written. 

The difference in the stream of token "a" versus the next token in the series "b" is primarily their attention scores and activation values! 

The exceptions are the input embeddings, and the new messages being written by token "b".

Honestly, I'm mostly unsure of what to do with that insight, but I think it's neat!


**"Prediction" vs. "Communication"**

I think it's worth noting that whatever communication may be going on within the residual stream, there is only one quality of it that matters in the end...

It needs to take the input word, and shove it around through vocabulary space until it most resembles the predicted output word.

(Or, more precisely, move it into the perfect spot in vocabulary space where the dot product similarity between it and every other word in the vocabulary captures the probability distribution of what the next word should be according to human language!)

We know that there are different subspaces in the residual stream--some which meaningfully affect the prediction, and others which don't. It seems reasonable to specualte that the non-vocabulary subspaces could be used by the model for "internal communication" and "metadata".

These two broad categories of data are all encoded into different subspaces of a single vector, and modifying one space can inadvertantly modify others.

Clearly Transformers are able to learn to handle this, but would they train more efficiently / perform better if we had two data streams--one for "metadata", and another "semantic", with only the latter used for prediction?

It would be fun to poke at the residual stream from this perspective, and try to tease the two apart!

**Path Expansion**

Finally, I believe a core insight of the Transformer Circuits framework is that each of these messages can be expanded into how it was created--the token's residual stream multiplied by one of the $W^M_i$ matrices. You can do a "path expansion" back through the model in that way, expanding each message into its parts. 

The same is true for the activation values and scores.

An interesting (albeit completely overwhelming) insight!


## Conclusion

The Residual Stream provides a mental model of the whole Transformer architecture, clarifies its additive nature, and offers insight into the broader context of the token messages.

I think it also serves as the gateway into interpretability research for those that want to explore further!

In the next post, I'll share my experience with a few probing techniques made possible by working with these vocabulary-space vectors. 

The techniques likely aren't novel, but I think we can apply them with more of an "educational" emphasis to create visualizations that deepen our intuition of Transformer behavior.
