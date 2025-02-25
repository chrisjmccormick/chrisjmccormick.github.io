---
layout: post
title:  "The Hidden Messages of Multi-Head Attention"
date:   2025-02-17 17:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1yL8t8FBrAOw1G1060ar2KZ6FU1whPxMP
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

In this post, we'll look at how a tiny bit of algebra suddenly opens up a wealth of insight. 

I want to clarify up front that none of the "rearranging" that I do in this post is intended to change how we _implement_ Attention. It's about _exposing underlying operations that are already there_--we just haven't noticed them because of the (important!) emphasis we place on finding computationally efficient algorithms.  

Let's see what happens when we prioritize conceptual clarity over GPU efficiency!

## The Hidden Structure in $W_O$

When we write out Transformer attention, the final step is to recombine the output of the heads using the matrix $W_O$:

<br/>

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

<br/>

The concatenation step allows us to perform a single, large matrix multiplication in order to combine the results of the heads and project them back up to the same space as the input embedding (the "model space").

I assume that's a good move computationally, but it obscures important insights about what is actually going on in the attention math.

## $W^O_i$

We don't call it out in the equations, but inside of $W_O$ are actually _independent, per-head_ output matrices, $W^O_i$. They are implicitly concatenated together and stored as one large matrix.

<br/>

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \text{Concat}(W^O_1, \dots, W^O_h)
$$

<br/>

We don't _actually perform_ that second concatenation, but you could say it's implied.

Concatenation is a bit of a bizarre operation for linear algebra--the resulting behavior is different than my normal intuition about what happens in a matrix multiply. 

Typically, we're multiplying every vector in one matrix against every vector in the other--i.e., all of the "items" in `A` interact with all of the ones in `B`. But it's different when you stack matrices like this.

The values in $\text{head}_1$ _only_ get multiplied against the values in their corresponding output matrix, $W^O_1$. They don't see any of the other parameters. 

The effect is the same as multiplying each of the corresponding smaller matrices against each other, and then summing the result:

<br/>

$$
\text{MultiHead}(Q, K, V) = \text{head}_1W^O_1 + \text{head}_2W^O_2 + \dots + \text{head}_hW^O_h
$$

<br/>

Seeing it broken out like this was surprising to me, because it highlights the fact that _each head_ is actually producing an independent output embedding,  and then these are getting summed together (without any weighting) to produce the end result.

On the left is the original illustration from _Attention is All You Need_, but I've modified it to include the names of the weight matrices, and to highlight the implied concatenation.



![Attention calculated per-head](https://lh3.googleusercontent.com/d/1yL8t8FBrAOw1G1060ar2KZ6FU1whPxMP)


On the right is an alternative version which is mathematically equivalent.

It's not as computationally efficient as the concatenation approach, so it's not how we implement it in practice, but it does make it a lot easier to explain the behavior of an attention head. 

We can illustrate and explain the _full_ behavior of a head without any reference to the concept of their being multiple heads. 


<img src='https://lh3.googleusercontent.com/d/1AfS0ML6LTFPWhsDNSkLtcifYiTpFGggD' alt='Single head attention' width='300'/>

The explanation for **Multihead** Attention then becomes, at least conceptually:

> _Every layer has a number of these heads running in parallel, and we sum their results._

Pretty straightforward! 

The concatenation operation becomes "a modification we do to run better on a GPU", and can be discussed separately. 



## Output of a Head


>Side Note: From here on, I'm going to switch to talking about having a single input embedding (our query) and a sequence of tokens to attend to (our keys).
>
>The fact that we can process multiple queries at once is another detail that's important for computational efficiency, but distracting for thinking more conceptually about how things work.

We have nice intuitions about what the _attention scoring_ part of a head does--we've demonstrated that the heads can identify semantic and syntactic relationships between words. It's figuring out "what words to pay attention to". 

Breaking apart the output matrix has helped me think more clearly, though, about what a head actually does on the _output_. 

We typically have two things that we might call the "output" of attention.


**Output of "Attention"**

The original equations define $\text{Attention}$ with the below equation, and later authors have adopted the variable $z$ for this step.

For a given input vector (and its query vector, $q_i$), the output of $\text{Attention}$ for head $i$ is:

<br/>

$$
z_i = \text{softmax} \left( \frac{q_i K_i^T}{\sqrt{d_k}} \right) V_i
$$

<br/>

$z_i$ is in the low-dimensional value space, $d_v$, making it harder to interpret.



**Output of "MultiHead"**

The other item we have is the result of $\text{MultiHead}$, which is given the variable $o$.

This is the output after combining all of the heads:

$$
o_t = \text{Concat}(z_1, z_2, ..., z_h)W^O
$$

$o_t$ is in model space, $d_\text{model}$, which is cool for interpretability. But it combines all of the heads into one embedding, so it's not really clear how each head contributes. 



**Output of a Head!**

After removing the concatenation step and breaking apart the heads, we can look at it with a little more granularity.

Each head is actually outputting its own independent embedding:

$$
o_i = z_iW^O_i
$$

Now for a given query, we can retrieve an output vector (in model space) to see what each head is returning!

This $o_i$ embedding never actually "exists" in the code, because the full-size $W_O$ step takes care of both computing the $o_i$ vectors and summing them together into a single $o_t$ in one big step.

But again--it's definitely there!

Now we can explain the output of attention as:

> _Each head produces a separate, full-size embedding (not a value vector!) which represents its overal contribution to the attention output._

That's a pretty cool insight, and something we should be able to exploit in investigating how the attention outputs impact the model. But it gets even better...

## Value-Output Relationship

Many illustrations of Attention follow the original in having the queries, keys, and values all pointing into an "Attention" block. 

I find this a little confusing, because the word "attention" emphasizes the part of the process which scores the tokens--the multiplication of query and keys to determine which tokens to focus on. 

The value vectors, though--they're a part of the overall process, sure, but they have more to do with the modifications to the embedding that we're going to make. 

It feels like the process ought to be split out to separate the **attention scoring** from the **embedding update**. 


<img src='https://lh3.googleusercontent.com/d/1vDHxMpEWoW3zOm1lSSSzIp9qaJyL_YJJ' alt='Separating VO process part 1' width='250'/>

Looking around online, there are definitely good illustrations out there which make this separation.

$\alpha_i$ is a square matrix of attention scores between every token, $T \times T$.

$V_i$ contains our tokens projected down into the small space of $\text{head}_i$, with size $T \times d_v$.

The standard equation for getting the output from those two things is:

<br/>

$$
O_t = \text{Concat}(\alpha_1 V_1,..., \alpha_2 V_2) W_O
$$

<br/>

But we can conceptualize it better by splitting this into:

<br/>

$$
O_t = \alpha_1 V_1 W^O_1 + ... + \alpha_h V_h W^O_h
$$

<br/>

And now we've stumbled into another way in which the computationally efficient definition has been obscuring something valuable.

Notice how the first form of the equation enforces a sequence of operations--the attention scores must be applied to the values first.

The second form lets us change the order, and compute the Output projections _before_ applying the scores.

<br/>

$$
O_t = \alpha_1 (V_1 W^O_1) + ... + \alpha_h (V_h W^O_h)
$$

<br/>

This creates a nice division of the two processes in attention:

1. The **queries** and **keys** are calculating the **scores**.
2. The **values** and **output** are calculating the **updates**.



<img src='https://lh3.googleusercontent.com/d/1-EyOcesnK-p7DjgJcA9rt19hMIQLMqwS' alt='Separating VO process part 2' width='250'/>

And now we've done something crazy. What is this $V_iW^O_i$ term?? It's been there this whole time, hidden.

We have:

Our tokens in Value Space: $V_i \in \mathbb{R}^{T \times d_v}$

Our output projection: $W^O_i \in \mathbb{R}^{d_v \times d_{\text{model}}}$

And now we have this new term, $M_i \in \mathbb{R}^{T \times d_{\text{model}}}$

$$
M_i = V_i W^O_i
$$

Each token in the sequence contributes a vector, $m_{i,t}$ to this matrix.
The final output of this head will be the sum of these $m$ vectors, weighted by the attention scores. 

<br/>

$$
o_i = \alpha_i M_i
$$



_How beautiful is that??_

Attention, summarized in two terms: the attention scores, multiplied by the messages.

# Messages

My original name for these $m$ vectors was **"modifiers"**. 

The output of attention, when you include the residual connection, will be:

$$
x^\prime = x + \sum_i{\alpha_i M_i}
$$

It's the input _modified_ by weighted sums of the $m$ vectors for all heads.

(Also, the variables for "adjustments" (A) and "updates" (U) don't feel available!)

I brainstormed with GPT and, after proposing a few options, this is what it had to say:

> "Messages" is my top pick because:
>   - It naturally describes the idea that **tokens send and receive information** through attention.
>   - It aligns with the idea of message passing in graph neural networks, which is conceptually similar.
>   - It feels intuitive without being too technical.

I think that's a really interesting framing!

> _Side Note: I wasn't aware at the time, but this "messages" language is standard in graph neural networks and has started to have some adoption in Transformers, particularly in papers bridging the two concepts. That said, the term has only been used to describe the value vectors, not their more interpretable model-space representations. I think adopting the language here will really help with our mental models of Attention!_

### Interpretation


What does it mean?

I've kinda assumed that, because the attention scoring can be mapped to linguistic relationships, that the "modifiers" must be updating the meaning of the embedding based on what the head finds. For example, for an input word like "run", there's a head looking for words to disambiguate its meaning, and then the head makes the appropriate adjustments based on what it finds. 

It may be instead, though, that the message in that situation is actually a signal to the FFN, and the FFN performs the actual shift in the word meaning.

That kind of counts as "messaging", but what GPT is hinting at seems more interesting. It's suggesting that the messages are a payload added to the token embedding, which will then be received by the attention heads in the next layer.


**In Decoders**

In a Decoder model, the messages can only be sent one way, making this a little easier to visualize.

The prior tokens are all set in stone--they can't receive any new messages. The only vehicle for passing new information from one layer to the next is the single word embedding that is our query vector.

If a past token gets a high attention score on some specific head, then the token gets the opportunity to add information to the query embedding to send to the next layer.

This payload could:
- Signal the FFN to do something, or
- Modify the query vector in a way that changes the attention scoring in some later head. 
    - An early head could tell later ones whether we're running for election, or running from the police (or both) so the later heads know what to focus on.



## Interpretability

The great news is, we now have some amazing new tools to answer those questions.

These messages are in _model space_, which means:

1. We can compare them directly to the patterns of the neurons in the FFNs. Which neurons react strongly to a message?
2. We can potentially compare them to the vocabulary embeddings--I've run some experiments doing this on BERT and GPT-2.
3. We can compare them to the patterns in the attention heads, using a new technique I'll cover in the next post.

And with all of that, we know exactly what word fired off what message!

# Conclusion

None of what we've looked at in this post actually changes or improves how Transformers operate. The Attention equations are written the way they are for a reason--efficiency on a GPU. 

The reframing of the equations I've done here is about clarifying and adding new insight, which will be great for future learners. 

That said, there are some more profound and practical insights that fall out of these discoveries as well. I'm excited to keep sharing what I've found, and to see where all of this takes us! 
