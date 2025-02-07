---
layout: post
title:  "How Reasoning Works in DeepSeek-R1"
date:   2025-02-07 8:00:00 -0800
comments: true
image: 
tags: Machine Learning, Natural Language Processing, NLP, LLMs, Reasoning Models, OpenAI o1, DeepSeek, DeepSeek-R1, DeepSeek-V3, Chain-of-Thought, CoT, TTC, Test-Time Compute
---

I’ve been really curious to know what’s actually happening behind the scenes when you ask OpenAI’s o1 model a question. From what they do show us, it seems pretty clear that the model is breaking the question down, tackling the problem in steps, reviewing its own work, etc. But considering how long the responses took to generate, and that the process was kept a secret, I assumed there must be something pretty exotic going on behind the scenes to make it all work.

With the excitement around DeepSeek-R1's competivie-with-o1 performance and open-sourced code, it was our chance to finally see how this whole “reasoning” process works in practice!

I dove deep into the papers and source code, found my answer, and confirmed it with some more knowledgable researchers.

At the foundation, _it's nothing more than asking the model to "think before you answer"_. That is, write out your reasoning before writing the solution. 

What sets the best reasoning models apart, it would seem, is just the usual--they were more effectively trained on the task!


---

_Why wasn't this obvious?_

This "tell it to think first" idea has been around for quite a while, so why'd it feel like such a confusing mystery--at least to me? 

Short answer: It's an active field of research! People have been trying various approaches, and I'll come back to these and why they still have merit.

---

### Chain-of-Thought (CoT) Reasoning

The general "think first" technique is referred to as **Chain-of-Thought** (CoT). 

It's well-illustrated by the prompt DeepSeek used during part of their training process:


----

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively, i.e., <b><font color="green">&lt;think&gt;</font></b><font color="navy"> reasoning process here </font><b><font color="green">&lt;/think&gt;</font> <font color="orange">&lt;answer&gt;</font></b><font color="navy"> answer here </font><b><font color="orange">&lt;/answer&gt;</font></b>. User: <font color="red">prompt</font>. Assistant:

----


That comes from Table 1 of the R1 paper, [here](https://arxiv.org/pdf/2501.12948); I added the coloring.

Seems like they must have done some prompt engineering to fine-tune that phrasing, but the key takeaway is telling it to write out its reasoning between `<think> </think>` tags. 


_How o1 does it_

OpenAI isn't quite as coy about their implementation as I thought, either. 

In their API guide for Reasoning, [here](https://platform.openai.com/docs/guides/reasoning), (in a section titled "How Reasoning Works", no less!), they explain:

> Reasoning models introduce **reasoning tokens** in addition to input and output tokens. The models use these reasoning tokens to "think", **breaking down** their understanding of the prompt and considering **multiple approaches** to generating a response. 

Since costs are a concern for developers, they also explain that:

> Depending on the problem's complexity, the models may generate anywhere from a **few hundred** to **tens of thousands** of reasoning tokens. 

### A Good Reasoning Dataset

A key challenge in training a strong reasoning model, it seems, is in acquiring a **good reasoning dataset**.

OpenAI doesn't let us see those thousands of reasoning tokens, otherwise we could "distill" with those to train a competitive model.

Similarly, although DeepSeek published their final model weights and explained their whole training process, they notably didn't share the high-quality reasoning dataset required to "cold start" the training.

(HuggingFace [launched](https://huggingface.co/blog/open-r1) a project named "Open-R1" to fill in the gap)







### Test-Time Compute (TTC)

The topic of **reasoning models** seems to be conflated with the term **Test-Time Compute** (TTC), meaning “extra computation done at inference time to improve reasoning ability”.


---

_CoT as TTC_

Chain-of-Thought technically fits that description--instead of trying to train the model to do all of its reasoning internally and **produce the answer directly**, we're allowing it to **spread out** the process over a larger number of tokens. 

Personally, I think the fact that reasoning involves _more compute_ seems rather besides the point and the wrong emphasis. 

I think that terminology is largely to blame for the confusion around this whole topic, but I'll let it go. 😉

---

**Trading Compute for Better Answers**

There are techniques which do involve deliberately using more compute power to get better answers, such as:

* Running the model **multiple times** to get **multiple answers** and picking, i.e., the most common one.
* More **complex decoding strategies** that involve exploring more possible generation paths.

[Asankhaya Sharma](https://github.com/codelion/) helped me make sense of this TTC landscape. He maintains a library called [OptiLLM](https://github.com/codelion/optillm) on GitHub, which implements a number of these strategies to apply on top of your existing model.

And according to @Fimbul (from our conversation on the Unsloth Discord [here](https://discord.com/channels/1179035537009545276/1257011997250424842/1335767339643310090)) these TTC techniques are especially effective at improving coding models, for example, so they're worth checking out!


### Conclusion

DeepSeek-R1's reasoning abilities, and likely o1's as well, boil down to being well-trained at writing out reasoning. 

R1 is still noteworthy, though, for being the first strong competitor to o1 and for DeepSeek choosing to publish their model and describe their training process.



_DeepSeek-V3_

Part of the commotion around DeepSeek this past month (January, 2025) actually ties back to their Decemer paper, where they published the "DeepSeek-V3" base model that R1 was trained from. 

V3 employed some interesting architecture changes that make it cheaper to train and inference (which lead to some political and stock market panic that the U.S. had lost its edge in AI!). 

I think they're some fascinating tweaks to the usual Transformer, and I plan to share more about them next!