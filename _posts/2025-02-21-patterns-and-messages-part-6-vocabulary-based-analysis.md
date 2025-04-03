---
layout: post
title:  "Patterns and Messages - Part 6 - Vocabulary-Based Analysis"
date:   2025-02-21 08:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1lxJaudzL_0ApmaFeQaDyw5GRXkp7Y6ZY
tags: Transformers, Attention Mechanism, Neural Network Architecture, Interpretability, Residual Stream, Multi-Head Attention
---

What had me most excited about the merged matrix perspective (and perhaps overly so) was that the patterns and messages are in **model space**, the same dimension as the vocabulary.

By applying the appropriate layer normalization to the vocabulary, it's possible to compare the patterns and messages to the vocabulary!

This is limited to "semantic" behaviors (i.e., relating to the meaning of words), and much of what's added to the stream must be other kinds of encoded information. Wherever there _is_ semantic behavior, though, it feels very satisfying to associate those vectors and matrices with actual words!

This technique can also be used on the word vectors in-between layers (the "residual stream"), as well as the individual or combined FFN output neurons.

> Side Note: A major caveat is that most recent models (Llama 3, Mistral, DeepSeek V3) do not use the same vocabulary vectors on their input and output, and I'm not sure if it's possible to make this work in that case.
>
> GPT-2, T5, and the recent Microsoft Phi-4-Mini are good examples of models that _do_ tie their vocabularies, though, and where this technique appears to work well.

I studied GPT-2 "small" (12 layers, 12 heads) in this post.


_by Chris McCormick_



## Contents



* TOC
{:toc}



## Existing Work



The field of interpretability research clearly has an abundance of insight about Decoder heads and GPT-2 in particular. 

An experienced researcher could probably look at the observations in this post and say "yes, the behavior you're seeing here is xyz".

I think it's been valuable, though, to arrive at some of these insights on my own, possibly with a different perspective, and vizualize them in my own way.

I'm looking forward to catching up on the research, and hopefully getting to illustrate more examples as I go! 

> As one glaring example of my knowledge gap, in this [blog post](https://www.lesswrong.com/posts/xmegeW5mqiBsvoaim/we-inspected-every-head-in-gpt-2-small-using-saes-so-you-don) the authors did some analysis of all 144 heads, and shared their notes on them in this Google [sheet](https://docs.google.com/spreadsheets/d/12HWFwUrs_W60pfjBOo6-CMLTfPG92XIbkr8_ANs1lA0/edit?gid=0#gid=0). _However_, I can't make sense yet of most of the terminology / shorthand they use in their discussion and notes! Definitely something to come back to.

## Approach

The general technique here is to look at the possible "meaning" of a vector by performing a kNN search between it and the vocabulary and looking at the top results. In cases where the top matches all share some clear property, we may be looking at an interpretable behavior. (Let's just be careful not to jump to conclusions too quickly!)

We'll apply this technique to various parts of the model over the course of the post.

### Input Sequence

All of the examples here come from running GPT-2 on the sequence

"The baseball player hit a home run"

With tokens:


```
<|endoftext|>, The, baseball, player, hit, a, home, run
```

No, I'm not a big baseball fan. (I am a scholar; I enjoy scholarly pursuits.)

What I liked about this sentence, though, is that most of the words have a unique meaning in this context, so it gives us the opportunity to look for ways in which the model is "contextualizing" these words.

I ran this sequence through the model and extracted pretty much everything--pattern vectors, message vectors, attention scores, residual stream, MLP activations.

I also compared every input and output neuron of the FFNs to the vocabulary to see what they most resembled (note that those results are always the same! They don't depend on the input sequence).


### Handling the Vocabulary

A few things to note about comparing to the vocabulary:

1. It requires some care in applying the right normalizations.
2. I filter out Byte-Pair Encoding (BPE) tokens to reduce clutter.
3. The closest matching tokens are often all very frequent tokens or very rare tokens, and I filter out these results.

Let me explain that third point.



**Filtering "High Bias" matches**

I learned from Neel Nanda's notebook [here](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=OnZjDTNu6ORV) that GPT-2 has an inherent positive or negative prediction bias for every token.

* The most **positive** bias is on the most **frequent** words like "the" or "and",
* The most **negative** bias is on many BPE tokens and very **rare** tokens like "RandomRedditor" and "SolidGoldMagikarp".

These high bias (positive or negative) tokens are often the closest matches to a vector!

A few guesses as to what these vectors represent, which could each be correct in different situations:

1. They have no meaning or information, and they're something like "harmless noise".
2. They have no semantic information, but are carrying other metadata or signals.
3. Presumably they often _are_ semantic and intended to align with tokens like comma, period, "the", "and", etc.

These results add a lot of clutter, so when the total bias of the results is above a threshold I just display "(high bias)" instead of the token matches.

I need to come back to this and see if there's a way to recognize number 3--when the vector is actually semantic--and not filter those results.

### Google Sheet

My analysis code writes the results into a table, which I then import into a nicely-formatted Google Sheet as my "visualization" tool. I've screenshotted specific examples to share in this post, but you can also explore the full results for yourself, [here](https://docs.google.com/spreadsheets/d/15x9bPSHBMXA2jK_1FR2_NYSHCseoXAq1uXNzm__eJwM/edit?gid=2119712321#gid=2119712321).

## Patterns

Let's check out the pattern vectors created by our tokens at every layer and for every head. (These examples come from the [Patterns](https://docs.google.com/spreadsheets/d/15x9bPSHBMXA2jK_1FR2_NYSHCseoXAq1uXNzm__eJwM/edit?gid=2119712321#gid=2119712321) tab of the sheet). 

We'll project each token onto a head's $W^P_i$ matrix to create its pattern vector $p_i$, and then do a similarity search between this and the vocabulary.

> Side Note: For this to work, you need to apply the layer's input LayerNorm to the entire vocabulary to bring it into the same space as the pattern vector.

Probably the most interesting observation about the pattern vectors of this "baseball" sequence is that... there aren't many meaningful results!

A few interesting observations, though:


### Self-Attending Heads




Some pattern vectors closely match the input token, implying that:

1. The token will **attend to itself**, which means that it will score **its own message** highly.
2. Whatever this message contains, it will probably get added strongly to the residual stream.
3. **Future** instances of the same word will attend highly to this pattern/message as well.

The below table shows the pattern vectors for a few heads in layer 0. Each column shows the "meaning" of the pattern left by that token at that head.


<img src='https://lh3.googleusercontent.com/d/1_XheqirWRIgcgt-okANyVTiMiFk2RhQH' alt='A filtered table showing heads that create pattern vectors which closely resemble the input word' width='1024' />

(Note that this is a good example of where my "high bias" filter is probably incorrect!)

Interestingly, the corresponding messages for these tokens aren't particularly interpretable. My initial assumption would have been that the messages also resemble itself.


<img src='https://lh3.googleusercontent.com/d/10Kn46-7dRKzJryTyzkwhhbLd5bLaAGK1' alt='A filtered table showing the message vectors for the heads that self-attend.' width='1600' />

My thoughts on the next example offer a possible explanation.

### Disambiguating Heads

**Baseball is Life**

The pattern vectors below are from the very first layer, so there is no context that it's drawing from yet. Head 0-0 just likes to put everything in terms of baseball!!

Specifically, it looks like _future_ baseball-related tokens will attend strongly to head 0-0 and pick up the messages left there by the tokens **hit**, **home**, and **run**.

(note that for 'player', the potential matches are more diverse than just baseball).


<img src='https://lh3.googleusercontent.com/d/1GTXOJ4QCkiFdVEDm7ZhZr8Xm43heilLG' alt='The pattern vectors for head 0-0, showing multiple words put into the context of sports' width='900' />

Perhaps this highlights a way in which Decoders handle their weakness of not being able to "go back and fix their work".

Imagine, for example, that the sentence was "The home run tied up the game."

The Decoder doesn't get to contextualize the word "home". It has to record its patterns and messages _without any context_, and never gets to update them.

So--what if it just leaves behind **multiple patterns** that will match the **different contexts**?

As if the token is saying: "If I'm baseball-related, read my message at 0-0. If I'm a house, read my message at x-x. If I'm a keyboard button, read..."

Here are the results for "home" for all 12 heads in layer 0:



| Layer | Head | Token | Most Similar Words to Pattern                                                                          |
|-------|------|-------|--------------------------------------------------------------------------------------------------------|
|   0   |   0  |  home | pitchers,  baseball,  Cubs,  outfielder,  clubhouse,  bullpen,  Astros,  shortstop,  pitcher,  catcher |
|   0   |   1  |  home |                          home,  homes,  Home,  HOME, Home,  **houses**, home,  **residences**,  **house**,  **indoor** |
|   0   |   2  |  home |                     stadium,  Stadium,  season,  England,  India,  UK,  Cameron,  bowl,  EU,  Scottish |
|   0   |   3  |  home |                 **investment**,  wooden,  home,  front,  arm,  opening,  economic,  Apple,  small,  **second** |
|   0   |   4  |  home |                               **Sweet**,  Jeff,  sweet,  New,  ref,  hot,  wooden,  pension,  wet,  moving |
|   0   |   5  |  home |                              home,  Home,  homes, home, Home,  HOME,  Homes,  **house**,  **homeowners**,  **New** |
|   0   |   6  |  home |                                                mining,  planet,  ship,  Water,  Enterprise,  carbon, � |
|   0   |   7  |  home |        **smart**,  **nursing**,  **tiny**,  super,  **funeral**,  progressive,  lightweight,  non,  health,  parenting |
|   0   |   8  |  home |     mortgage,  investor,  investors,  buyer,  buying,  sales,  purchase,  energy,  buyers,  homeowners |
|   0   |   9  |  home |                                                                                            (high bias) |
|   0   |  10  |  home |                           home,  Home,  homes, home, **backyard**,  **indoor**,  **house**,  Homes, Home,  houses |
|   0   |  11  |  home |                                    <\|endoftext\|>, He, She, Most, More, There, 76, .$, Some, Although |

Head 9 may be outputing garbage, and head 11 as well (it must have just barely missed the cut off for my "high bias" filter!)

Otherwise, it does seem like a fair amount of contextualizing going on?


**What Might This Mean?**

In our "The home run" example, the token for "home" never gets to clarify for itself what it is. But there could instead exist multiple pathways through the model that handle it differently depending on the context.

Maybe the message left here at head 0-0 is a kind of **signature** added to the residual stream so that, in later layers, the stream matches the correct-context patterns for "home"?

We might be able to test that idea by doing something along the lines of comparing these different "context signature" messages with the pattern vectors for "home" in subsequent layers.

### Repeating Heads



Head 6-6 seems to be creating patterns which include some of the same terms over multiple future tokens.

Note "hit", "March", and "flight" as some examples.


<img src='https://lh3.googleusercontent.com/d/1rn2DnM38-N6BkhYQ3Ovgt44csoJHUW1s' alt='The pattern vectors for head 6-6, which show various words repeated over multiple tokens' width='900' />


I'm reluctant to speculate on what behavior this is showing exactly...

One thing I find interesting about it, though, is that this behavior requires **multiple layers** to execute.

Some kind of signature, whether semantic or otherwise, is being picked up as a message in an earlier layer, so that the word then shows up again in the pattern vector for head 6-6.


**Visualizing Patterns and Messages Together**

In a later section I have a visualization that shows the pattern and message vectors side-by-side so we can (try to) see what is sent when an interesting pattern vector is matched.

For now, let's take a look at some message vectors on their own, since this makes it easier to notice when a head might have a consistent message-writing behavior across multiple tokens.

## Messages

These examples come from the [Messages](https://docs.google.com/spreadsheets/d/15x9bPSHBMXA2jK_1FR2_NYSHCseoXAq1uXNzm__eJwM/edit?gid=2126454947#gid=2126454947) tab of the sheet.

### Repeating Messages



One of the more interesting behaviors we can observe in the message vectors is some type of repeating behavior.

On the final layer, head 11-3 is producing a very similar baseball-context message over the whole sequence.

<img src='https://lh3.googleusercontent.com/d/1pYddsaz0HNEyuJB8nk9oDVxbaZS_RCD6' alt='A table showing the messages for head 11-3, which show how the context of baseball is carried forward by this head' width='800' />


This behavior can also be seen in heads 4-8, 7-1, 9-8, 10-0.

In the "labeled heads" [sheet](https://docs.google.com/spreadsheets/d/12HWFwUrs_W60pfjBOo6-CMLTfPG92XIbkr8_ANs1lA0/edit?gid=0#gid=0), 11-3 is described as simply "long context" which makes some superficial sense, but I'll need to go learn what they mean by that more specifically. 

What I can say here is that:
1. This head is in the final Decoder layer (layers are 0-indexed), which seems to be have a particularly high influence on the meaning of the residual stream (i.e., it plays a large role in deciding what the next token should be).
2. Future tokens which match the head 11-3 patterns will pick up these semantic sports / baseball messages.
3. Some mechanism in an earlier layer must be carrying forward a signal that tells 11-3 to output baseball messages.

In the "residual stream" table that I'll discuss further down, we can see that:
1. The corresponding patterns at 11-3 don't have semantic meaning.
2. None of the tokens (in our tiny example) acctually end up attending to this head.


**Aliteration / Shared First Letter Names**

This is a fun one. Let's get the more bizarre behavior out of the way first...Head 8-11 creates messages containing the names of people and places which start with the same letter as the input word. (Or, in the case of the token "a", the names all include the letter 'a').

* Baseball --> Buck, Buff, Bryce
* Player --> PA (pennsylvania)
* Hit --> Hernandez, HIT?, Hoffman
* a --> Philadelphia, Hanna, Sean, Fay, Shay, Ryan, Katie
* run --> Robinson, Rodriguez, Rod

Peculiar!

<img src='https://lh3.googleusercontent.com/d/14H-QEtPJf9Ml5cvRhBwE1rN3zdGX9HkZ' alt='The message vectors for head 8-11 showing repetition as well as names with the same first letter' width='900' />

**Identity of Past Input Tokens**

Less bizarre--several of the messages for head 8-11 strongly match the input word, such as "baseball", "player", "hit", and "home".

This appears to be relatively common. Heads 6-9, 7-10, 9-9 are other strong examples.

Something interesting about attention is that tokens don't actually get to "see" the prior tokens! There's nothing _inherent_ in the attention mechanism that lets the current input see what words came before it.

We often illustrate attention as attending back to prior tokens, but I'm realizing that's a little misleading. The input token is only attending back to the (low dimension!) **messages left behind** by prior tokens.

Heads like this one could provide a way for future tokens to see the "identity" of prior tokens. But note that this is only if they match the corresponding pattern vector--and this is another instance where, in our tiny example, none of the tokens actually match this head.

**Prediction**

One other noteworthy observation: head 9-0 seems like it could be involved in predicting future tokens. Many of the matching words seem like good candidates for either the next or an upcoming token.

That would need to be reinforced by more examples, though, before I'd claim it.



**All Together**

Next, we'll put everything together in one giant table which will allow us to see:
* The messages in the context of their patterns
* The attention scores, which tell us:
    * What patterns each token actually ends up matching to.
    * What messages each token actually ends up receiving.
* The possible semantic meanings of the FFN input and output neurons.
* Which output neurons ("neural memories") are added to the stream, and how strongly.



## Residual Stream

In the previous post on the residual stream, I illustrated how the input vector, by the time it reaches the output, can be viewed as the weighted sum of every message and every neuron output (the dimensions in the illustration come from Llama 3 8B and assume a full sequence length of 8k tokens):

<img src='https://lh3.googleusercontent.com/d/1_duz3CEV6DUV27NB31IeeKZUKLX8XSZ_' alt='The residual stream drawn as a stack of the neural memories, messages, and input vector, with scores and activations, and example dimensions' width='500'/>



**Let's build that table and look inside!**

Here we'll stick with GPT-2 and only have 8 tokens. 

Such a short sequence means we only have ~1k messages (144 heads x 8 tokens).

The FFN neurons dominate the table size, with ~36k of them total (12 layers x 3,072 neurons).

It's all pulled together in the [Stream](https://docs.google.com/spreadsheets/d/15x9bPSHBMXA2jK_1FR2_NYSHCseoXAq1uXNzm__eJwM/edit?gid=1642348242#gid=1642348242) tab of the sheet.









## Attended Messages

Before going on to the FFNs, let's look again at the patterns and messages, this time seeing them side-by-side, and including the attention scores to see what patterns were actually matched / what messages were actually picked up.

**Table Layout**

There is a lot going on in this table, so it might take a bit to process. 

Here is what head 0-1 looks like as an example, which is the most strongly (aggressively!) self-attending head for this sequence.

The columns on the right represent each "input token" and the attention they gave to each context toekn. 

On the left, you can see the "meaning" of the pattern and message vectors that each token left behind at this head.

For head 0-1, the pattern vectors are synonyms for the input words, so it makes sense that the self-attention score would be so high.

<img src='https://lh3.googleusercontent.com/d/1MbZyA-sZzaeO5QobNH6MMj7qbD_gypK4' alt='A table showing all of the tokens and their attention to the pattern and messages for each context token at head 0-1, which is strongly self-attending' width='900' />

**Semantic but Illegible?**

The messages aren't high bias, so they _might_ be semantic, but they're certainly not "legible".

The reason they could still be semantic is that, as a loose example, if you calculate `hit - player`, is there any reason to think that the **delta** would necessarily match specific vocabulary words that make sense?

It's also possible that an attention head could communicate through a **composite** of messages, and the semantics only emerge when we combine them. 


**Attending to BOS**

The attention pattern at 0-1 is very unique. 

A _much_ more common pattern is to see most of the attention placed on the beginning of sequence (BOS) token. 

<img src='https://lh3.googleusercontent.com/d/1_E75vG8eMw-Ay1uXHtuufA35lq8CMbZJ' alt='A table showing all of the tokens and their attention to the pattern and messages for each context token at head 4-9, which has some interesting attention and messages but also demonstrates how tokens predominantly attend to the first position' width='900' />


An idea from the "What Does BERT Look At?" paper is that this type of behavior (dumping all of the attention somewhere) is a way for a head to perform a "no-op" when it knows it's not relevant.

That's a very satisfying explanation when looking at these tables. It gives the impression that Attention boils down to:

1. Most heads produce patterns which are heavily biased towards position 0. 
2. A pattern match has to outperform this bias in order to steal away some of the attention weight from position 0.
3. This means the default behavior of the heads is to (strongly) add whatever message is left by the first token to the stream!

> Side Note: I have some plots which show this bias towards position 0 nicely. The technique involves using the pattern projection matrices $W^P_i$ to construct patterns for the PEV vectors on their own. I'm hoping to share this in another post!

**BOS is Constant**

Something remarkable to me about this behavior is that GPT-2 wasn't trained with a BOS token.

That feels like a strange choice given that:

1. When a token is at position 0, all of its contributions to the residual stream are constant--they don't depend on any context, and will always be the same. 
2. Whatever those contributions are, they seem to be made heavily (especially for early tokens).

It seems like what we put at position 0 might be pretty important! 

I suppose this bias to position 0 means that maybe its contribution to the stream is fairly consistent regardless of which token we put there.

I wonder if models would perform better, though, if they could rely on something more stable being at position 0? 

We could also try fine-tuning _just_ the BOS token to see if there's something to put there that better aligns with what the model saw during training on average.


**The Missing Trifecta**

For looking at the attention heads in this table, what I was most hoping to see was a kind of "trifecta" where we see examples of the attention scores, pattern contents, and message contents all aligning in a logical way.

This seems to be pretty rare, though! Usually it's only _either_ the pattern or the message which is "legible".

The below table shows one of these rare examples in the first row. Several tokens attended to the word "player" via head 5-10, which produced a pattern resembling words like "teammates" and "scored", and a message which resembled words like "athlete" and "injury".

I also bolded the word "who" because, as we'll see in a later section, that's the word the residual stream most resembles going in to layer 5.


<img src='https://lh3.googleusercontent.com/d/1KzjjES9FJ5QIjFLJ3iSWuL2PzOVRft0F' alt='A table showing a couple examples where one token attended to another and where the attention scores, pattern contents and message contents all feel meaningful' width='900' />

**Non-Semantic Signatures**

I think the second row is more interesting, showing the pattern and message for the word "baseball" at head 9-8.

The message has clear semantic meaning, but the pattern looks like garbage. _And yet_, the attention pattern looks meaningful!

So what is creating the high dot product with this "garbage" pattern? This seems like evidence that the pattern vector is capturing some kind of non-semantic signature which is present in some of the tokens.

You can find many similar examples of this (tokens attending in an interesting way to a "high bias" pattern).


**Attention vs. FFNs**

I've been focused here on what we can uniquely visualize using the "patterns and messages" perspective of attention, but this same technique can be directly applied to the FFN neurons. 

While the messages have an inherent low dimension (because of the Value-Output decomposition), the FFNs don't have this restriction. 

The paper "Neurons are Key-Value Memory Cells" found that the FFNs are involved heavily in prediction, meaning they play the biggest role in changing the semantics of the input vector into the output vector. 

Since they're so semantic-heavy, let's try looking at their neurons!

## FFN Neurons

Here's how we can apply our vocabulary analysis to the FFNs.

First:

1. Each neuron has two length 768 vectors which are its input and output weights.
2. The activation for a particular neuron is based on how similar the residual stream is to its input weights.
3. If a neuron activation is non-zero, then the output weights will be added to the residual stream, weighted by the activation.

The input and output weights of _some_ neurons have clear semantic associations!

By comparing the **input weights** of a neuron to the **vocabulary**, we can see what words in the vocabulary it might be **looking for**.

By comparing the **output weights** to the **vocabulary**, we may be able to see what semantic information will be added by that neuron to the **residual stream**.

Below are a couple fun examples.

Here's how to read it:
* **Layer**: The Decoder layer number
* **Neuron**: Each FFN has 3,072 neurons, so this is just the index of a sepcific neuron in that Decoder layer's FFN.
* **Most Similar Words**: I think of the FFN in terms of pairs of input and output neurons. Both are represented by a length 768 vector which can be compared to the vocabulary.
    * Note that these word similarities never change! They're not dependent on the sequence.
* **Activations**: For each token in the sequence, the number shown is the dot-product between the word vector and the input neuron, with the GeLU activation function applied.
    * The heat map color coding goes from:
        * 0.0 - White
        * 0.5 - Green
        * 1.0 - Red
    * GeLU allows activations to drop slightly below 0, but not much, and the positive end is unbounded.
    * When the value is exactly 0.0, I replace it with "--" to clean things up a bit.

<img src='https://lh3.googleusercontent.com/d/164hPJ7IlFyJ61uTczNRBJBEMaaW4A2cV' alt='A table showing the activation values for two specific neurons, along with the most similar words to their input and output patterns, showing a neuron which predicts player for baseball, and another that predicts balloon for pop' width='900' />


First, neuron 10-2361 is highly correlated with one of our specific word transitions: baseball --> player!

> Side Note: In NLP, these are often referred to as "bi-grams", a pair of words which are likely to occur together.

Second, neuron 0-1259 associates "pop" with "balloon", and I thought that was adorable, so I included it.

Again, we're looking at model weights here, so this "what word mapping does this neuron handle?" never changes.



**Legibility**

Overall, here are some things I'm noticing:

* Many of the input neurons resemble groups of synonyms or related terms. This is especially true in the first layer.
* Far fewer output neurons seem legibile.

Similar to the patterns and messages, I'm not seeing many neurons where both the input and output make sense.

The most common case where they _do_ both make sense are ones where the input and output seem to be very similar in meaning.

Finally, even when the input neuron resembles a clear topic, it may fire for seemingly unrelated terms. It's possible that we need to look at more than just the top 10 most similar words, though.

Something I found a little more intriguing were the neurons that respond to the "BOS" token.


**Beginning of Sequence**

Something that hadn't occurred to me before is that whatever the first token in a sequence is, everything about it (patterns, messages, activations) is _always the same_.

When generating text, it's common practice to prepend the "Beginning of Sequence" (BOS) token. The models don't always define a specific BOS, and we use the end of sequence token instead. In GPT-2, this token is decoded as `<|endoftext|>`.

The below table shows some FFN neurons which activate _extremely_ strongly for the `<|endoftext|>` token, and _not at all_ (or very little) for any other token.

<img src='https://lh3.googleusercontent.com/d/12oXn2qz64ZlG6t-jZcCwDK_X_LM6eVEU' alt='A large table highlighting the FFN neurons which respond only to BOS and activate very highly for it, and the output neurons match high bias tokens' width='900' />


This seems fascinating. In layer 2, whatever is hiding in the output of Neuron 666 (ok, seriously?), this vector gets added to the residual stream with a weight of **61.9**. 

This Layer 2 "Devil Neuron" gets added to the residual stream with a higher weight than anything else in the entire residual stream.

Also in layer 2, neurons 1825 and 3034 also receive similar huge weights. I really want to come back and explore those three vectors!

Now that we've proven that GPT-2 is fundamentally evil, let's look at how the combined FFN outputs and messages gradually evolve the meaning of the residual stream through the layers. 

## Token Evolution

By comparing the word vectors in between layers to the vocabulary, you can watch as the meaning of the vector gradually evolves from the input word to the predicted output word, which is pretty neat!

You can find this on the [Token Evolution](https://docs.google.com/spreadsheets/d/15x9bPSHBMXA2jK_1FR2_NYSHCseoXAq1uXNzm__eJwM/edit?gid=1394655586#gid=1394655586) tab of the sheet.


<img src='https://lh3.googleusercontent.com/d/1lxJaudzL_0ApmaFeQaDyw5GRXkp7Y6ZY' alt='A table showing how the tokens in the sentence The baseball player hit a home run evolve in their meaning through each layer' width='900' />


Each row shows the "meaning" of the residual stream at the input to the specified layer. The bottom most row, (labeled "predicted") shows the meaning of the final output vector.

It's interesting to see where the words tend to change most in meaning (i.e., where the model tends to make and change its output prediction)--typically in the middle layers, and then again in the last two. 

Note that the final layer (layer 11, 0-indexed) changes the prediction more than half the time! (It makes me think of someone second-guessing themselves on a test and changing their answers at the very last second).


**Accuracy**

It can be difficult to visualize how well / poorly it's doing at predicting the prompt. The below printout helped me a little.

Input Tokens:

```
<|endoftext|>
 The
 baseball
 player
 hit
 a
 home
 run
```

<br/>

Predicted Tokens (inside the vertical bars):

```
<|endoftext|>| The|
<|endoftext|> The| first|
<|endoftext|> The baseball| team|
<|endoftext|> The baseball player| who|
<|endoftext|> The baseball player hit| the|
<|endoftext|> The baseball player hit a| home|
<|endoftext|> The baseball player hit a home| run|
<|endoftext|> The baseball player hit a home run| in|
```

It only predicts 3 out of 7 tokens correctly.

**The Challenges (Insanity?) of Encoding Prompts with Decoders**

It's pretty illuminating to watch it process a prompt...

We tend to create our Decoder illustrations showing the input words and the output words as though it neatly maps one to the next.

In reality, the word vector starts as the input word, transitions through various meanings, and quite commonly at no point in the pipeline does the word resemble the correct prediction.

The problem is that next token prediction is impossible. If I tell you to complete the sentence "The baseball player"--how are you supposed to guess the correct answer?

It's a little better when we look at the top 10 matching words because the correct answer may be among them (you can also view the image full size [here](https://lh3.googleusercontent.com/d/1USIzhxZN2eJVkqFUjGO13LIIVsAiAZPB)).

Note though, how the model predicts "The baseball player | who|", and at no point does the token resemble the correct answer: "hit".



<img src='https://lh3.googleusercontent.com/d/1USIzhxZN2eJVkqFUjGO13LIIVsAiAZPB' alt='A table showing the ten most similar vocabulary words to each token residual as it moves through the 12 layers of GPT-2' width='900' />


Whatever pattern matching is happening, and whatever information the tokens are leaving behind in the cache, is mostly based on an intermediate prediction, and one that's often heading in the wrong direction.

(During text generation, things are more sensible. Whatever the model predicts is the correct answer.)



**Messages are Cached, Not Tokens**

I pointed this out earlier, but it seems worth emphasizing again here.

* We can't actually "see" the prior tokens.
* We can only read the (low dimension!) messages they left behind.

Position information also isn't explicitly preserved if we're using Position Encoding Vectors (PEVs)--it can only be stored in the low dimension patterns left behind. (RoPE solves this, though!)




**How Does it Even Work?**

My speculation is that Decoders compensate for this by being packed full of redundancy--repeatedly untangling the garbage prompt encoding rather than being able to rely on it.

The example we saw earlier around disambiguating the word "home" may be a concrete example of this.

**Encoder-Decoder Models**

I'd love to better understand the challenges in training Encoder-Decoder architectures to outperform Decoder stacks.

I've heard training data is part of the problem, but it seems like we have a pretty good way to generate input + desired output training samples now, yeah?


## Conclusion



Exploring the model in this way, with lists of words to read and interpret, can be very fun, and very distracting!

I know that there's a big risk here of jumping to (possibly false) conclusions around what a particular data point means, and claiming behaviors without adequate evidence.

It does seem like a valuable approach, though, for:

1. Corroborating and illustrating behaviors which have been more rigorously demonstrated.
2. Possible new insights, particularly if the approach is more granular than in existing work.

I hope I get the opportunity to come back to this with a more disciplined approach!
