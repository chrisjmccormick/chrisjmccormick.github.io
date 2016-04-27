---
author: chrisjmccormick
comments: true
date: 2013-03-29 23:08:29 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/03/29/ucf-lecture-6-optical-flow/
slug: ucf-lecture-6-optical-flow
title: UCF Lecture 6 - Optical Flow
wordpress_id: 5471
---

The following post contains my notes on lecture 6 of Dr. Shah's computer vision course:


[http://www.youtube.com/watch?v=5VyLAH8BhF8](http://www.youtube.com/watch?v=5VyLAH8BhF8)




Optical flow is the motion of a single pixel from one frame to the next.




Optical flow is used for:






	
  * Motion based segmentation:

	
    * Identifying objects in the scene based on their movement versus the stationary background (the object is moving, but the background is not).




	
  * “Structure from motion”

	
    * Retrieving 3D data about an object from its movement.

	
    * Recall the video in the first lecture of balls representing a person’s joints--our brain can recognize this as a person based on the motion.




	
  * Alignment

	
    * Video stabilization (removing camera shake)

	
    * Video from moving camera (like a UAV, again recall video from lecture 1)




	
  * Video compression

	
    * Apply displacement to one image to synthesize the next, encode any difference/uniqueness.







### **Brightness Constantancy Assumption**





	
  * 


Horn & Schunk




	
    * 


Horn is a famous MIT vision researcher, Brian Schunk was his PhD student.




	
    * 


They came up with the basic optical flow concept.







	
  * 


Brightness constancy assumption: If you take the frames very close together (30fps), a pixel will move slightly but it’s brightness will not change by much (because it’s only been 1/30s).







****![](https://lh6.googleusercontent.com/YFB5D66NYywjXNwEk1x6s45IZt_OgBYBjj4INHleRmkKx4b-FOXQn7lQQ7SO8lh_0UGK81lXO9C0U2ca3pRkjY0qEborzFt9iLZT-it45y95fMBd6p0zoJOhXA)****




[10:30]





### 




### 




### **Taylor Series Approximation**




We're going to take the taylor series approximation of the above equation.




I found the equations on Wikipedia to be slightly clearer than the ones Dr. Shah uses.




![](https://lh3.googleusercontent.com/jW7X0BRK5SjNkPJtpU2-ee1cz65_yCyk-Jn0eo4X3zR46LlPUAICgXEVqecs1wK5GeEh2Mx_3xZB52R5p1AZWe94z9rS4COtztXugC1QG4IyrrIS14IQeGab)






	
  * I(x + dx, y + dy, t + dt) = The brightness (intensity) of a pixel at its new location in frame 2.

	
  * I(x, y, t) = The brightness (intensity) of a pixel in its location in frame 1.


This equation only shows the first two terms of the taylor series (only the first derivative). The other terms must not be significant.

This equation just states that the intensity of a pixel in frame 2 is equal to it’s intensity in frame 1 + a small change.

	
  * 


The first term, dI / dx * delta x




	
    * 


dI / dx = The rate of change in intensity relative to the x axis.




	
    * 


Multiplied by the small change in x gives you a small difference in intensity.








We are making the assumption that the pixel brightness is constant, so we’re assuming:


![](https://lh6.googleusercontent.com/gQhCEXe14SfJVOmevlFS97MCEeIefpiMgiZ9qq9pC5yuwFmx44YM3Ppjh-qQ1ijO1jLnPRycpsrf_hgjX0dWwc9nlpOdHCYpCP1oaD03bYcD-noVRln-7O-y)


Subtituting this into our Taylor series approximation above, we find that:


![](https://lh5.googleusercontent.com/X-BDhBDKztn6yVz0C6UvSEv-pvhjUO6HB_VQDSPtHX2FkSyJJNr2QzdqE1nhaajY6bW_aI0V0_C2U2SYwvS3OomUUM9LqFq21j09sWsmP21ruIXWjYlZVv8E)


The above equation simply states that the change in pixel intensity is 0 (between the first and second frames, and with the small amount of motion of the pixel).

We said that the pixel is moving a small amount, dx and dy, between the two frames. And the two frames are separated in time by dt. So we can find the pixel's velocity in the x direction as dx/dt, and the its velocity in the y direction as dy/dt.

To do this, we divide the equation by dt and we get:


![](https://lh3.googleusercontent.com/wYlijFNuyIb6ZVS7aifmfil4bYT-vVOz31F0PIfxPqQTSlDKtKwG_5YFZCEiVKiD97QeOghU8cYBcHRQ4KkXvUuAdIWr_LV0_pDdZElIxY-j3aPpAH9vgV0N)




We write the x component of the velocity as Vx and the y component as Vy. We also remove dt / dt, leaving us with:




![](https://lh5.googleusercontent.com/evERhtXoKvVBN1LdhwgcTEHE50y-saDWWOWBq9cmFY_UeLG8rKOTh5PwPXA_l_6qNWczGOtQNZxL_odz5RUGQDdqYmGLDlY9CecktVGhECLtXRRk2iaG2r9x)




Finally, another notation for “the derivative of I with respect to x” is Ix, so we have:




![](https://lh5.googleusercontent.com/0ttMMdd4Bl_6JCHxus0-Z6--4ctrbnpnW9ASfmrXaOVeUYl8qab_G1V7KPvU2r9akVRt-N-w--ndLKWbihr1MES-Sp1aqP09RWHlsHa2157-FRWopMtxe596)




In the video lecture, he uses ‘u’ for Vx and ‘v’ for Vy




It’s important to know that given two sequential image frames it is possible to compute Ix, Iy, and It. However, Vx and Vy are unknown. Which is simply to say that for a given pixel in the first frame we don’t know where it will end up in the next frame.





### Normal Flow & Parallel Flow




![](https://lh5.googleusercontent.com/JAV8jnBLRB2sXdNId60T4sYnln4mn1wm3ulAa5STnVVcVpfXXUI35vsndc4ZwvT17kIYNErbiGifuY73BYLM2Cve8MtlplRhyKv187JLg1BDR2_Ajxsb3v26Kg)




[15:00]




We have two unknowns, u and v. But we can write this as a linear equation, and we at least know that the u and v will lie somewhere on the line as shown above.




We do know that the vector (u, v) is equal to the vectors p + d






	
  * d is called “normal flow”

	
  * p is called “parallel flow”

	
  * 


We can derive that d in the above figure is equal to:







![](https://lh5.googleusercontent.com/-F2I8KgFv9k6WcrJZStweTCgPKzkIbhVqxSlWBPagWhTd_8CiM3edN7hvr3ZlxdcMPKtvVgDzwUDLhsuS_oJjyeOBlgG91yqFo2powJowaPsfHvldm6-WPru5Q)




[16:31]




So we know ‘d’, but we dont’ know ‘p’.




Note: He never comes back to this notion of 'p' and 'd', so I'm not sure why they're included in the discussion. It seems sufficient to understand that Vx and Vy are unknowns, and that we need to add an additional constraint to the problem in order to solve for them.





### Adding a Smoothness Constraint




[17:40]




Horn & Schunk treated this (not knowing ‘p’) as an optimization problem.




![](https://lh3.googleusercontent.com/iZNqkRfrjbJOfDi5kz1Wat1XSttmBP7xEDmqFh-mKGj2P9qQwkkPQmg435nCPi886d4uGFQ7ZUZya-oHSkMUqrNea-OmbpeDPJprlqmnRVIWGIEuLTmDXd9JZA)






	
  * The double integral means we’re doing this for every pixel.

	
  * The first term:

	
    * This is the brightness constancy function, which we’ve said is ideally 0.

	
    * It may not be exactly 0, though, there may be a small error, so we square the error and get a small number for this term.




	
  * The second term:

	
    * This is adding a smoothness constraint, the idea being that if the object is moving, all of the pixels in the object will have similar motion.

	
    * Object boundaries will be a problem, though (this only kind of makes sense to me).

	
    * Take the derivative of the velocity. If the change is smooth, then these derivatives will again be small values. (Again, I’m not very clear on this)




	
  * Ideally both terms should be small. So the idea is to minimize this function (find its minimum). He decides to not go into a lot of detail on the math

	
    * At a peak or valley in a function, the derivative is equal to 0. So to minimize a function, you take its derivative and equate it 0, then solve the equation.

	
    * If you take the above equation and take the derivative with respect to u and equate that to 0, and also take the derivative with respect to v and set that to 0, you get the following two functions:







![](https://lh6.googleusercontent.com/oSbZuXKebvKMAaInCpLrUO4yUH2_Y_D6J0IaVArACfqm9_vrNrIWQQB064nD3ZVS_x-DhaY4HiwGgymTL40LBbEuSlsbWBlUF4Rvyg2Z2meHS0VJV230YvggOA)






	
  * The second term we recognize as the Laplacian, which is equal to the second derivative of u with respect to x plus the second derivative of u with respect to y.




****
![](https://lh3.googleusercontent.com/xveKLI5E1qH0Ic04CM4vLRhJPcKSlykZEdlZV9FnPMahPc-f5iBEYIkRWiHDKnZ0NA9S0BmZMUNpm-4SeYl7vLY8_LDgsZJ6TUWyGaJdC-wMbWUsro0Tj3bDbw)****




[22:27]





### Derivative Masks




He shows the Derivative masks used by a researcher named Roberts. They’re similar to the masks we saw in the lecture on edge detection. The ft mask is a new concept, though.




****![](https://lh6.googleusercontent.com/khZf2y_db5To2NFdIz6J8ajxg_92YTEumoApr7LT18ScTsRL0L_t_-XbBJ6bIa82ACapxoBto-gDTfin1yu0WQPUe8CWTAc3f_q7QSCm8uu7WxpoJuke-LyDnA)****




[24:37]





### Laplacian Mask




This is a new concept.




****![](https://lh3.googleusercontent.com/s6wdHmfafXPWq_zLpkVBOptvuF87nYLD5PwMpTJyGPyI-apC4ChiG23Awk83eJSRPh1cGAzRCFjKiND2F8gsTvNpzxMcUlv7IaZhUq95qeG_s0XA2KphTwTOmg)****




[25:38]




Take the average of the four neighbors (multiply each neighbor by 1/4) and subtract that average from the value of the central pixel.




Using this approximation for the Laplacian, we can re-write the earlier smoothness equation:![](https://lh4.googleusercontent.com/_N4oBog7IuDvRvjO3OtmF6OFIYUBu8fwefN_g_xCnOp_rreTsDsdUx4eCtLjGIgLojU49MzTWs_TZSk3KUGsUAVs4877o5lOJ6BZ9zWiKWAxVhGjGvmFZB8M1Q)




[26:20]




Now we have two equations for u and v, and we have all of the other values, so we can solve for u and v as:




****
![](https://lh5.googleusercontent.com/F_y0jjRROTaH4eYJmVHB3U6gQ7gn2SbRfalgxR4YnIbXmtUuMFwcnxqCrHOXfbwtkfxrnMDSvfRe6EoTDbBVW5YPF0lj-guNMisGZsguBf69m7NF9hkwdNggBA)****




[27:15]




There is a step now that I don’t understand. You start with initial values of u = 0 and v = 0, then “iterate”. Does iteration refer to going over subsequent frames? I don’t think so, the implication seems to be multiple iterations of a single frame.





### Lucas & Kanade (Least Squares)




[31:00]




Optical flow dates back to 1980s, many more advancements have been made since.




Kanade - famous researcher, Lucas was his student.


****
![](https://lh6.googleusercontent.com/sIQlu00tcl3Edw47-gHk6cpbQNNS6c0xKh4MFcROPUIK_eTOfYAwiU6vFB41uEz9G_XR3lIaQMavBJ7cwvMFOpfio2mCCp91GHk1TC49bAoT086PQE6DGYSETQ)****


[33:20]






	
  * 


Take a 3x3 window, and assume that all of the pixels have the same x and y velocity.




	
  * 


Now we have 9 equations and two unknowns.




	
    * 


The equations can be represented as a matrix as shown.




	
    * 


We representl the matrix on the left as ‘A’.










****
![](https://lh5.googleusercontent.com/C4AWXx7n-k8QQOI2IcX_whGSmQtgSzdRgqEAv6QEp5kXpza-M36I396icHZYzlDCOP-OFdeCh1FO2drmTLEArNtMBMiYRqI8hrkdZBOHzAmdODbOcYuID26seA)****




[35:13]






	
  * 


We need to solve for ‘u’, but it’s not possible to divide both sides by A.




	
    * 


It’s not possible (?) to take the inverse of a matrix that isn’t square.




	
    * 


But If you multiply the 9x2 matrix by its 2x9 transpose, you get a 2x2 matrix. That allows to separate out the ‘u’ term, giving us the equation above.







	
  * 


The three ‘A’ terms in the final equation are referred to as the ‘pseudo-inverse’, apparently this is a common concept.







### Least Squares Fit




[37:20]


Once we have u and v, ideally the optical flow equation at each pixel in the window should equal 0. In practice, there will be some small error at each pixel.


The optimum value for u and v will minimize this error. This is expressed by the following equation.




![](https://lh4.googleusercontent.com/jI6YqKwsO4_HmonC3R1SLCOOQjFujbranOEe-S-g3u56f32JHg953Q6eYrh7cDiY07cJXPtmYjuEab9OhtuRfsqufasPCfEB7DaG9PmSLEBS0_0T0cUywA4xHw)




Note that squaring the error means that larger errors will contribute an exponentially larger amount to the total than smaller errors.




To minimize these functions we take the derivative, once with respect to u and again with respect to v, and set both equations to 0.




****
![](https://lh4.googleusercontent.com/ZS3m08bDPFn3zu-KX5NdW2EYx02eDCBrt-yeFo996N8a_bojCa3yAcmC_tm0z90hXh7gascpBjxfeIeUCUB6bBLOra9eGp8cas_JCYtwAV6s18uHZ0l8CrJLxw)****




[36:27]




The summations can be expanded to:




![](https://lh5.googleusercontent.com/kRIE1f233czVJNJyXriRNoLroN-jJtT1Oz7gHkRglgqGtgWXvVIVhiV6dHi_5XsCmn7m463Im_iF9jjYHHDhBa5q7GtKe9Men_ngacRu_5UXsmW76_EG8npyeQ)




Then rewritten in matrix form as:




![](https://lh6.googleusercontent.com/i8vUOHcUzXyjsY0h47OAzdzrIFxWPVEQzc2KgsGezjyzBsRYQS1cYW1xSD7KxSYWMgrdQv0JeSmR8Xty5UOI-fiO9ZuTOh9JkiO94xnd5LRWu7ywKcPtu9x2pA)




This 2x2 matrix can be inverted, giving us:




![](https://lh3.googleusercontent.com/T5HgZLnMApg1LJo6DLW5VXw_dEtGKaUBFAEuqkMbwxhnXQ_TjtHeS9wkaZlg3VvoPb0lSHIH6VbWI1TMFm6Vz7c1h7yrRH-BLZRr_rifJvnsAftBeCFdkk1lcA)




Expanding this with the definition of a matrix inverse, and writing the equations for u and v separately, we have:




****![](https://lh3.googleusercontent.com/NHoKxShQOkQ2sniQXIZD30UZnYs5fNwZNmbtnjTudWN5CzhjUZOh_QTB7ZnX0MazXIw2KvWAQtSAAz0M3Ilf6h6AO_phVGiOduGpWHdEfMGg5euqaS05n9Q2cA)****




[41:03]





### Lucas-Kanade With Pyramids





	
  * 


The Lucas-Kanade method fails when the motion is large, for example if a pixel moves 15 pixels from one frame to the next. The reason Dr. Shah gives is that the derivative mask is only 2x2 pixels.




	
  * 


The solution to this is something called pyramids.




	
    * 


Make multiple scales of the image.




	
    * The 16 pixel motion will become a 1 pixel motion if you scale the image down by 16.





