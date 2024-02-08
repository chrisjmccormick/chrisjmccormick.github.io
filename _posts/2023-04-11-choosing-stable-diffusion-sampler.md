---
layout: post
title:  "Choosing a Sampler for Stable Diffusion"
date:   2023-04-11 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1vCaEOyVoWcwBrfObAcdL9eSrAwVBgdKq
tags: Stable Diffusion, Sampler, Euler, Ancestral, DPM++ 2M, Sampler Comparison, AI Art, Techniques
---


I’ve studied the samplers a bit and done some of my own experiments with them, and I’ve arrived at some tentative conclusions for what to do with them. 


Essentially, there are already so many different settings to play with (not to mention the prompt!), and I don’t think exploring the different samplers will net you much. It seems better to simplify this choice, especially as you’re just starting out.


So, to boil it all down, I’d recommend experimenting with both “**Euler**” and “**DPM++ 2M**”, and I think that’s good enough! 


## Contents


* TOC
{:toc}


## Euler and DPM++ 2M


These two samplers produce similar results, so if you’re exploring a particular prompt and seed that you like, you can try out both of these samplers to see what you think.


Here’s a quick look at what these each produce. In the below experiment, after finding an appealing seed, I tried a range of step counts, CFG values, and both samplers.



[![Euler vs. DPM cyberpunk girl 2](https://lh3.googleusercontent.com/d/1EOoyN2bXAZ-1t2vxejLiZ3fg6IAjTbuY)](https://lh3.googleusercontent.com/d/1EOoyN2bXAZ-1t2vxejLiZ3fg6IAjTbuY)


I used a prompt I found on Lexica [here](https://lexica.art/prompt/7f5d6d2c-580e-4247-86ce-77e3b9e3c488).


## Solving Ordinary Differential Equations (ODEs)

My understanding of the math behind the samplers isn’t quite complete, but I know that “Euler” is the most primitive approach to the task that the Sampler is tasked with. 


“Euler” refers to “Euler’s Method” for “Solving an Ordinary Differential Equation (ODE)”. (Whatever the heck that means, right?! 😜)


ODE Solvers are guess-and-check methods for solving equations that can’t be solved more directly. 


They estimate what direction the correct answer lies in, and then take a step in that direction and re-evaluate. How large of a step they take is a trade-off between getting there faster and not over-shooting.


I don’t think they necessarily find the exact right answer, they just try to get “close enough”. 


My understanding is that the other samplers are just different algorithms that employ clever tricks for making better estimates so that they can arrive at a decent answer in fewer steps.  


This is why you’ll hear samplers talked about in terms of reducing the number of steps you need to run Stable Diffusion for, and how it’s possible that the sampler alone could have this impact.


Besides Euler, “Heun” and “LMS” are other classic ODE solvers you’d find in a textbook. “DDIM” and “PLMS” are the “originals” used with Stable Diffusion, but seem to have been replaced by DPM++.

(Note: A lot of these insights came from reading [this article](https://stable-diffusion-art.com/samplers/), which offers some insight into each of the samplers).       
 
## Faster or Just Different?

Again, you’ll often hear samplers discussed in terms of their ability to reduce the number of steps required to produce a good result. 

That sounds very appealing, because running Stable Diffusion is slow and expensive, and it’d be great if (A) we could get what we want in, e.g., half the time, or at the very least (B) preview seeds more quickly.


In my experience so far, though, I don’t think either of these are true in practice.


_(A) Faster Results:_


The problem with (A) is that, as you can see in the example earlier in the post, DPM++ produces results that seem to have high-detail at all step counts, but this isn’t always preferable.  For that image, I think Euler at 25 steps and CFG 7.0 might be my top pick, despite being lower-detail.


Really, once you’ve found a seed that you like, I think it makes sense to explore it with various combinations of settings (again, as in the earlier example). And that means running for lots of steps regardless.


_(B) Faster Previews:_


For previewing seeds, it’s not obvious to me that DPM++ at 15 steps is a much better preview than Euler at 15 steps. 


One of the biggest things that seems to influence my interest in a particular seed is how structurally sound it is. The style may be really interesting, but if, e.g., body proportions are wrong, I’m likely to pass on it. At 15 steps, Euler produces softer / less-detailed images, but they also seem to have fewer defects!


_Sampler Style_


In the end, I think the samplers seem more useful for their differences in style rather than the more practical benefits they might provide.


## Ancestral Samplers


You’ll notice in the sampler list that there is both “**Euler**” and “**Euler A**”, and it’s important to know that these behave _very_ differently! 


The “A” stands for “Ancestral”, and there are several other “Ancestral” samplers in the list of choices. Most of the samplers available are _not_ ancestral, and produce roughly the same image for a given seed and settings (as we saw with **Euler** and **DPM++ 2M**), but the ancestral samplers will give you different outputs. 


They produce “interesting” results, but I’ve ultimately decided _against_ using them.


Apparently the ‘ancestral’ approach involves adding some additional randomness at each de-noising step, and this has an interesting implication around varying the step count. Instead of refining an image, different step counts will actually produce notably different images!


Here is that same example, this time comparing Euler to Euler A. 





[![Euler vs. Euler A](https://lh3.googleusercontent.com/d/15Ir-YOl4LNRAQC3Xki_Zg7Ip4yjScAl-)](https://lh3.googleusercontent.com/d/15Ir-YOl4LNRAQC3Xki_Zg7Ip4yjScAl-)


Certainly the Euler A results are very striking, but:


You can also get striking results just by running more random seeds. 
Note how inconsistent Euler A is across the different settings, which can be problematic for refining your result.
As cool as Euler A’s outputs are, if you look closely at them, I think they all have significant details that are irrational or have poor structure. 
For example, one of my favorites in the Euler A grid is the very bottom right (50 steps and scale 13), but look at how her bangs appear to be tucked into her eyepiece…


Stable Diffusion already struggles with creating rational details, and in my limited experience, Euler A seems to suffer from this more than the non-ancestral samplers. 


If your **goal** is to generate images that are as **defect-free** as possible, then the ancestral samplers seem like a dangerous _distraction_. 


If you’re just looking for inspiration, or have the skills or other tools to fix the defects, then I could see the ancestral samplers having more value!   


I’m typically after the former goal (few defects as possible), so I’ve decided to avoid Euler A.


## “Karras” Versions


In Auto1111, you’ll notice that most / all of the samplers have a second copy with the word “Karras” appended to it. 


This refers to Tero Karras, whose paper here argued for a fundamental improvement that these versions implement. 


Some people use the Karras versions by default, but I haven’t noticed a clear advantage, so I’m sticking with the non-karras versions for simplicity.


## Additional Examples


_Landscape_


“Rivendell”


[![Eulver vs. DPM Rivendell](https://lh3.googleusercontent.com/d/17WdAbF-bJRSofXN-CUSwMBveYhPcyceD)](https://lh3.googleusercontent.com/d/17WdAbF-bJRSofXN-CUSwMBveYhPcyceD)


The prompt is from [here](https://lexica.art/?q=fantasy+landscape&prompt=c06ccb12-0b44-40e2-84cb-e8ec3d92941d), and the full prompt and settings are at the end of the post.


## Appendix: Example Settings


All examples were created with Stable Diffusion **v1.5** using Automatic1111 on Colab with this tool.


_Cyborg Woman Portrait_


```
Portrait of a beautiful caucasian young cyberpunk woman cyborg ninja, third person, D&D, sci fi fantasy, intricate, richly detailed colored , art by Range Murata, highly detailed, 3d, octane render, bright colors, digital painting, trending on artstation, sharp focus, illustration style of Stanley Artgerm, Steps: 25, Sampler: Euler, CFG scale: 13, Seed: 1702264368, Size: 512x704, Model hash: 4c86efd062, Model: model
```


I found the prompt on Lexica, [here](https://lexica.art/prompt/7f5d6d2c-580e-4247-86ce-77e3b9e3c488).


_Rivendell Landscape_


```
rivendell from lord of the rings matte painting by yanick dusseault and dylan cole, artstation, 4 k, insanely detailed,
Steps: 25, Sampler: Euler, CFG scale: 7, Seed: 1189946030, Size: 768x512, Model hash: 4c86efd062, Model: model
```


I found the prompt on Lexica, [here](https://lexica.art/?q=fantasy+landscape&prompt=c06ccb12-0b44-40e2-84cb-e8ec3d92941d)
