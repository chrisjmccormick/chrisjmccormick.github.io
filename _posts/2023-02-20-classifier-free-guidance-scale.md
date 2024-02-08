---

layout: post

title:  "Classifier-Free Guidance (CFG) Scale"

date:   2023-02-20 8:00:00 -0800

comments: true

image: https://lh3.googleusercontent.com/d/1NojH3pillX5_46MHZ-e5m6jTwzZRmHCe

tags: Stable Diffusion, AI Art, Tutorial, CFG, Classifier Free Guidance

---

The Classifier-Free Guidance Scale, or “CFG Scale”, is a number (typically somewhere between 7.0 to 13.0) that’s described as controlling how much influence your input prompt has over the resulting generation. 

It’s easy to misinterpret that explanation, though, and to expect the wrong thing from this parameter, so let’s look at CFG scale in more detail.

## What Does “Guidance” Mean?

Stable Diffusion generates art by gradually “restoring” a noisy image into a piece of artwork. (It operates under the assumption that there’s a painting hiding under all that noise, and it’s trying to uncover it). It does this gradually over a number of steps–making small adjustments to the image each time.

To decide what improvements to make to the image at each step, it looks at the noisy image and tries to puzzle out what it’s looking at.

For example, the below image is still pretty rough, but you and I can clearly see that it’s supposed to be some kind of alien or monster playing a guitar, and we could fix it up if we had the artistic skill.


![Noisy alien guitar](https://lh3.googleusercontent.com/d/1FtaSiBxOS_wVpAcXgPqraKxZf0b4oqGG)


Of course, in addition to looking at the image, it’s also being **guided** by your **description** of the image to generate.

Making visual sense of the below image is really hard without knowing that it’s supposed to be a set of stairs leading down into the ocean! 


![Confusing stairs into water](https://lh3.googleusercontent.com/d/1yspl1CmCjDPyFXY9ySha2FQedzHvXNHi)


This is what “guidance” refers to in “Classifier-Free Guidance”--the image generation being _guided_ by the text description.

And the CFG “**Scale**” refers to the ability to increase or decrease the amount of **influence** the text description has on the image generation.

This can sometimes improve the quality of the generated result. In the below example of “Bob Ross riding a dragon”, it’s not till a scale of 13 that we get something reasonable.


![Bob Ross drag at different scales](https://lh3.googleusercontent.com/d/145qU11UzxcMw_BcBaNQPjyBoBAOBrFDA)


## What Does “Classifier-Free” Mean?

And what the heck does “Classifier-Free” mean? The inclusion of that term is unfortunate, I think, because it’s just a reference to an older technique for guiding the image generation that’s no longer relevant. It’s similar to if, instead of “Electric Cars”, we called them “Gasoline-Free Cars”. 🤦‍♂️

> For more technical readers, here’s my understanding of the classifier technique: Instead of generating form a prompt, you could only specify a category of object to generate, like “dog”, “cat”, “car”, “plane” (i.e., the ImageNet categories), and the model used a standard image classifier to evaluate the progress and help supervise the diffusion process.

## What to Expect from CFG Scale

When you’re having trouble with a generation, it’s tempting to try getting more and more specific in your description, and to feel like you’re dealing with an **obstinate child**. You think, “_surely_ the model understands what I’m asking for, and it’s just being _stubborn_ and not listening to me”, so you crank the CFG hoping it will **start obeying** you.

In reality, if it doesn’t seem to be understanding the intent of your prompt (even after you’ve generated lots of examples and used various wordings) then it’s probably just beyond the model’s current abilities. 

In the below example, the prompt subject is “A painting of a horse with **eight legs**, standing in an apocalyptic wasteland”. I really like the seed, but upping the CFG scale doesn’t seem to do anything to increase the number of legs!


![Eight legged horse attempt](https://lh3.googleusercontent.com/d/1tj8zneqhD35BHBIWgCjusNSSwj6oSF5z)

The best approach I’m aware of currently for exercising more control is to use image-2-image generation.

## Another Source of Variety

In practice, I think it’s best to simply view the CFG scale as another way to vary the results of your generation.

Once I’ve found a prompt and seed that I like, I like to use the technique of generating a grid of images to explore different combinations of CFG values and step counts, as in the below example (click the image to view a full resolution version). 


[![Wizard grid](https://lh3.googleusercontent.com/d/1lAVuhic6gXzoofmwZWY1UhmdRzkL1E90)](https://lh3.googleusercontent.com/d/1lAVuhic6gXzoofmwZWY1UhmdRzkL1E90)



FYI, Automatic1111 has this feature in the “scripts” section: 



![Grid tool in Auto1111](https://lh3.googleusercontent.com/d/1f0hsFEt0z8ljzy7x3xiW0ZxxssbR5qq0)



(I’ll have to provide a tutorial on Auto1111 at some point!)

## How CFG Scale Works

The remainder of this article is probably most interesting to more technical readers interested in understanding the implementation, as well as some insight into why the technique is not all that effective in practice.

When you give Stable Diffusion a prompt to generate, it actually generates _two images_ in parallel–one guided by your prompt, and one not (technically, the second image is guided by an _empty_ prompt). 

The difference between the two is considered to be the influence of the prompt, and we scale that influence up or down by multiplying it with the CFG scale.

### Two Artists

To understand the intent of this, imagine you took a talented artist named “Tim” and cloned him. We’ll call his clone “Ted”. (We’re doing this so that Tim and Ted have identical minds).

We ask both Tim and Ted to restore this image, but only Ted gets to know the description.

Tim is going to make adjustments purely based on what he can see in the image. “I think this looks like an astronaut, so I’m going to take things in that direction.” 

But Ted is going to use a combination of what he sees, and the description he’s been given. “I can tell there’s a person standing there, and it looks like an astronaut, but the prompt says it’s a guitarist, so I’m going to go in that direction.” 

(I made Tim and Ted clones so that they have an identical visual interpretation of the image). 

Here’s where the _scaling_ happens… At each “iteration” of working on the image, we can look at the difference between Tim and Ted’s suggested changes, and lean more heavily to one or the other. 

In practice, we always lean towards Ted’s suggestion more, and the scaling factor just determines by how much.

### Amplifying Bad Influence

If Ted (who has the description) _doesn’t really understand_ the prompt the way you’ve written it, or isn’t familiar with your subject matter, or maybe just doesn’t have the skill to create it (SD seems to struggle with more complicated imagery), then _amplifying Ted’s influence on the result isn’t going to solve the problem_.

I have to imagine that setting the scale to, e.g., 1,000 would just “break” things and give you garbage. The tools limit you to more reasonable values, though. Auto1111 goes up to 30:



[![High CFG examples](https://lh3.googleusercontent.com/d/1ZCwP-el4BJYJbioCa0Boapi_XyqnVo8S)](https://lh3.googleusercontent.com/d/1ZCwP-el4BJYJbioCa0Boapi_XyqnVo8S)



It’s absolutely worth experimenting with, though! 

### The Original Purpose of CFG

In fact, the real purpose of the CFG parameter is that, In the witches brew of math that was used to **train** stable diffusion, apparently this guidance scaling technique was _critical_ for getting good results during _training_.

Stable Diffusion was trained with this scaling factor set to some value (I’ve tried to find out what exact value was used, but so far no luck! The closest guess I’ve got is that the CompVis library sets the cfg to 7.5 by default), and this greatly improved how well the model performed on the training task.

Then, the ability to tweak this parameter at “inference” time (i.e., when we’re using SD to generate art) is actually just a secondary benefit.

## Appendix: Examples

```
Prompt: “bob ross riding a dragon, model pose, ultra realistic, concept art, intricate details, highly detailed, photorealistic, octane render, 8 k, unreal engine. art by artgerm and greg rutkowski and alphonse mucha”, 
Negative Prompt: “”, Seed: 1442287716, Euler A, 30 steps, 512x512 
```
```
Prompt: “A painting of a horse with eight legs, standing in an apocalyptic wasteland, trending on art station, by greg rutkowski”
Negative Prompt: “jump, jumping, leaping”, Steps: 20, Sampler: Euler a, CFG scale: 13, Seed: 2405405571, Size: 512x512
```
```
Prompt: "full face epic portrait, male wizard with glowing eyes, elden ring, matte painting concept art, midjourney, beautifully backlit, swirly vibrant color lines, majestic, cinematic aesthetic, smooth, intricate, 8 k, by ilya kuvshinov, artgerm, darius zawadzki and zdizslaw beksinski ",
Negative Prompt: ""
Seed: 1718067705, Sampler: Euler, Size: 512x704, varied steps & cfg
```




