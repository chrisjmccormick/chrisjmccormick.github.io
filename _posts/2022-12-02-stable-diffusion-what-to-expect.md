---
layout: post
title:  "What You Can Reasonably Expect from Stable Diffusion"
date:   2022-12-02 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1LZ9J-TKM6xRS7qMKXV1u7DX25SPkacf1
tags: Stable Diffusion, AI Art, Prompt Design, 
---


I think AI art **inspires awe** in us because:



1. It's incredibly **imaginative**.
2. The **artistic technique** is masterful.
3. It was created by an **AI**. 

Unfortunately, what it so masterfully generates also tends to be very **incoherent**.

In my experience, you should probably decline any offers for a ride in a stable diffusion fighter jet... 😜


![Stable Diffusion Fighter Jet](https://lh3.googleusercontent.com/d/1iV3ALD-fHYTVG-1LTLWg-fCDuDH8cOBY)


Browsing libraries of generated imagery, the most popular (and successful) subject to generate seems to be **portraits**.

![Symmetrical Portraits](https://lh3.googleusercontent.com/d/1J4aMHF-879G6-nL4u8645iX8bJPo4yfu)


(From [here](https://lexica.art/prompt/5408d538-1057-4ba0-be08-2b65279e107b), [here](https://lexica.art/prompt/601d3548-5c86-491f-8175-0020edc801b6), and [here](https://lexica.art/prompt/6417498c-787d-4b92-af46-a26df5e541c6).)

A head-on perspective, with a lot of **symmetry**, seems to work best. There are strong examples of other perspectives as well, just less common:

![Portraits](https://lh3.googleusercontent.com/d/1MQgy_xBsqDJtxr0I0IqJ7h-CvgwTuoO0)


(From [here](https://lexica.art/prompt/9e3fe4d4-92ab-4d5f-a076-2c40fd54c80e), [here](https://lexica.art/prompt/29ad931b-78d0-4429-8036-3a987ab49358), and [here](https://lexica.art/prompt/937830f5-19ad-46d0-9592-c885f883bf60))

If you start going below the shoulders, though, many of your generations will be “ruined” by unrealistic posing of limbs or incorrect proportions. 

Most critically, though–Stable Diffusion v1.5 seems, for all practical purposes, **incapable** of generating hands and fingers, or the correct interaction between a person's **hands and an object**. 

![White Knights](https://lh3.googleusercontent.com/d/1LZ9J-TKM6xRS7qMKXV1u7DX25SPkacf1)

(From [here](https://lexica.art/?prompt=a5dec1f5-5fb9-47b0-bdae-4706cd848232))

![Holding Objects](https://lh3.googleusercontent.com/d/1rqa5Ji7S1_TjQpZk0eUNj1GryuoZEDfw)

(From [here](https://lexica.art/?q=holding+a+book&prompt=d033c544-8ee5-4317-a024-d805605b5922), [here](https://lexica.art/?q=a+hand+holding+a+cup&prompt=3da96c57-5fda-4262-959d-0ac78d6f3609), and [here](https://lexica.art/?q=eating+a+burrito&prompt=10d96a80-e158-4537-8146-728fa9d36684))

That's pretty discouraging, because most the time when I have an idea for something I want to generate, it's a **scene**--a **subject** in a **setting**, interacting with one or more **objects**. For example, a “blacksmith forging a sword in his workshop”...

![Blacksmith working](https://lh3.googleusercontent.com/d/1a5sM-kIMuO26iCazab_AecWvEKmXqiN6)

(From [here](https://lexica.art/prompt/24151852-85ea-4e9c-a890-343442ed746e))

It seems clear to me that you’re just not going to get a polished image straight out of the model on something like this (no matter how much you play with the prompt or settings). 

But there's hope! There are a number of fancy techniques out there that we can try, such as:



* We can use **inpainting** to regenerate a specific object in the scene
* With **outpainting**, we can first generate a subject that we like, and then expand outward to create the setting. 
* Providing a **starting image** can help us dictate the layout of the scene.
* **Compositing** can help us blend in replacements for parts of the image.
* **Fine-tuning** can help it generate a particular subject or style more reliably.

And, while I don’t want to become a full on graphical artist or photoshop expert, I’d be willing to pick a few of those tricks here or there to get what I want. 😊

I’ve kicked off a new [YouTube series](https://www.youtube.com/playlist?list=PLam9sigHPGwNLOWrCGtJDcBo7oBuJNLy6) to explore all of the above! I’ve started out providing an introduction to the absolute basics of AI art generation, since there’s still plenty to learn about **prompt design** and the purpose of the different **settings** before getting to the more advanced techniques I listed. 

I also plan to publish blog posts here and there to cover specific topics. Stay tuned!
