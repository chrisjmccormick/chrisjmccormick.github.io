---

layout: post

title:  "Steps and Seeds in Stable Diffusion"

date:   2023-01-11 8:00:00 -0800

comments: true

image: https://lh3.googleusercontent.com/d/1SKyqrglsqcdIiybAhGSt5YwDRkhOKhFn

tags: Stable Diffusion, AI Art, Tutorial, Step Count, Seed

---


In this series of posts I’ll be explaining the most common settings in stable diffusion generation tools, using DreamStudio and Automatic1111 as the examples.

This first post will cover the **steps** slider and the **seed** value, and then further posts will cover the “cfg scale”, and “sampler”.


## Steps

Let’s start with the step count.

In DreamStudio, this is labeled “Steps”, with the description “How many steps to spend generating (diffusing) your image.”


![Inference steps setting in DreamStudio](https://lh3.googleusercontent.com/d/1t8SeAzjRYDERnXspzNoZ5JoV8idCWr8T)

In Automatic1111, it’s labeled “Sampling steps”:


![Step slider in Auto1111](https://lh3.googleusercontent.com/d/17xsDvkcIHjDDtEqcPYAmfESkAbP_hta6)


With the following tooltip: “How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results”. That’s a pretty good summary, but let me give you a little more insight into how it works.

[Update 4/7/23]: A new version of DreamStudio has been released with a re-designed interface! The screenshots in this post come from the original version, which is still accessible [here](https://legacy.dreamstudio.ai/dream). The settings I'm explaining are still there in the new version, just laid out differently.   

### What Happens in a “Step”

Stable Diffusion generates images by starting with **random noise**, and then, using your text prompt as guidance, gradually “**de-noises**” the image. (It treats the initial random noise as though it were actually a very noisy image, and it’s trying to **recover** the _supposed_ original image, using your text description to inform it).

In the illustration below, the image on the far left is our **initial noise**, and the image on the far right is the finished result. I ran this generation with 25 steps; the center illustration shows the state of the image at each step, and you can see how the image is gradually getting cleaned up.

![From noise to artwork](https://lh3.googleusercontent.com/d/1vWAL0IUFNWrT3Q5IbULIOmr4svZL9-GW)

No matter how many steps you use, the process _always starts and ends the same way_–you start with random noise, and end up with a completely de-noised image.

What the “steps” setting _actually_ controls is how many steps we **divide** this process into. So a **higher count** divides the process into more steps, and each step will make a **smaller change** to the image. 

If you use what would be considered a relatively **low** number of steps, you’ll find that the images tend to be less detailed and contain more **flaws**. 

In the below examples, the result at 5 steps is a complete mess, and 10 and 15 both have uneven irises. 

![Step count 5 to 20](https://lh3.googleusercontent.com/d/12c205yfoh272Qgv-rWyCz29ZSCUx1Gwt)

![Step count 20 to 50](https://lh3.googleusercontent.com/d/1SKyqrglsqcdIiybAhGSt5YwDRkhOKhFn)


There does appear to be some correlation between the number of steps and the amount of detail, but I’d argue that beyond a certain point you’re really just getting different variations.

It also seems that you _can_ go **too high**. In this example, it looks like 100 and 150 steps starts to reintroduce some problems.

![Overdoing the steps](https://lh3.googleusercontent.com/d/1wqAFN1v9Gib66Sxii0kAUb19ZlYu_pPx)

Side Note: The effect of the number of steps can also depend significantly on the choice of “sampler”--something we'll look at more closely in a few posts. The illustrations above were generated with "Euler A", which is the default selection in Auto1111.

### Common Misconceptions

There are a couple misconceptions I’ve seen around the effect of the step count that I think come from over-simplified explanations. 

The **first misconception** is that increasing the step count means “**continue refining** the image longer”. 

You might run for 10 steps, look at the result and think, “I like that, but I’d like to see it more cleaned up, so I’m going to try running it for 5 more steps by setting the step count to 15.” 

In reality, if you look at my earlier example and compare 10 steps vs 20 steps, it’s pretty clear that 20 steps is not just a “more refined” version of 10 steps. They’re  different images!

Remember, it’s always the same process, the step count is just _how many intervals_ you _divide_ the process into. The different subdivisions will cause the model to follow slightly **different paths**. 

The **second misconception** is that higher step counts will always yield higher quality results, and we’ve seen how that’s not the case (in our comparison of 50, 100, and 150 steps). 


### Steps & “Compute Cost”

There’s a more practical impact of the step count as well: It affects **how long** the image generation takes–more steps will mean you have to **wait longer** to see your result. 

The Stable Diffusion “model” is a _dizzyingly_ large Neural Network. For those familiar with the concept, it contains roughly **1 billion** parameters! The number of mathematical operations required to run a _single step_ is on the order of _trillions_.

Tools like DreamStudio run the model for you on “GPUs”--the **graphics card** that you normally use for playing video games. 

However, the GPUs they typically use are tailored towards this type of math-heavy, scientific computing (This is Nvidia’s “Tesla” line of GPUs, such as the Tesla T4 or Tesla A100). These specialized GPUs are outrageously expensive! 

Below is a **Tesla A100**; its original MSRP was **$32k**! 🤯

![Tesla A100 product photo](https://m.media-amazon.com/images/I/411mNK+mP-L._AC_SY580_.jpg)

When you use a paid tool like DreamStudio, they are going to **charge** you (using a “**credits**” system) based on the number of steps you run. The compute cost is also directly related to the **size of the image** you want to generate, so higher image resolutions are going to cost more credits.

![Credits in DreamStudio](https://lh3.googleusercontent.com/d/1lakzPtdKM0mS6JLojG_NlpCZ00WH-Xin)


### Preview Images using Few Steps

Given the longer weight times and higher cost of running more steps, a strategy some people use is to first generate images with a **low** step count, just to get a general sense of each of the images. Then, for any results that look promising, they go back and **re-run** these for a **higher** number of steps to get a quality image.

The way to “re-run” a particular image is using something called the “seed” value, which we’ll look at next.


## Seed Value

The “seed” is a number like `145391` which, if you specify it to Stable Diffusion, will always generate the **exact same image**. (provided you keep everything else the same, too) 

![Manual seed setting in DreamStudio](https://lh3.googleusercontent.com/d/1vFngK1yT2qHdC55_prXrd7ZGjGh9UbZb)

This allows you to try and tweak an image that you like. You can, for example, make **small changes** to the **prompt**, or change the number of inference steps, and get results that all look like **variations** of your “base” result. 

(For example, to generate my earlier example images demonstrating the effect of the step count, I used the _same seed and prompt_, but varied the steps).

Note that there’s _no relationship_ between how close the seed numbers are and how similar the images will be. If you change the seed number by 1, you’ll get a completely different result. For example, the below images have seeds that differ by 1 (…201, …202, and …203). 

![Nearby seeds are different](https://lh3.googleusercontent.com/d/1PvJBEEYfoVMAe-IEENMyVC7UjvgXDUFn)

They have some similarity, of course, because they all used the same prompt.


### Aside: How do Seeds Work?

For those curious…

In computers, **random number generators** aren’t _truly_ random. For a given starting point (“seed”), the generator will always spit out the exact **same sequence** of “random” numbers! 

Creating the initial random noise involves setting the different pixels to random values. But as long as the seed is the same, the noise image will be **identical**! 

One more thing… With Stable Diffusion, if you specify “**-1**” then it will randomly choose a seed / seeds for you… using another random number generator to select the seed values! 😂

So how do we make sure _that_ generator gives us a random seed?? 🤔

When you don’t provide a seed, my understanding is that the generator will base the seed on something that will always be unique, such as the _current date and time_. 


### Setting the Seed

I’ll conclude by just pointing out where to locate and set the seed value in these tools.

In **DreamStudio**, if you **hover** over the image, you’ll see the seed value here:

![Find the seed value in DreamStudio](https://lh3.googleusercontent.com/d/1zhIB4ynyNI-kUhrZxDPMbVsvIyWZipFB)

Clicking on the seed value seems to copy it to the clipboard and the seed field for you.

In order to actually _set_ the seed value in DreamStudio, you must **set the number of images to 1**. Otherwise the seed field is hidden from view.

![Manual seed setting in DreamStudio](https://lh3.googleusercontent.com/d/1vFngK1yT2qHdC55_prXrd7ZGjGh9UbZb)


In Automatic1111, it’s more straightforward:


![Set the seed value in Automatic1111](https://lh3.googleusercontent.com/d/1m6At4jC4ZQAr1a0cP7MD27BitmLMK2zF)


You can find the seed used to generate an image in Auto1111 by looking at the output window, or the filename, or by opening the image in the “PNG Info” tab.

