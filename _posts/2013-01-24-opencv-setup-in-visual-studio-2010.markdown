---
author: chrisjmccormick
comments: true
date: 2013-01-24 18:45:01 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2013/01/24/opencv-setup-in-visual-studio-2010/
slug: opencv-setup-in-visual-studio-2010
title: OpenCV Setup in Visual Studio 2010
wordpress_id: 5366
tags:
- OpenCV
- OpenCV Setup
- Visual Studio 2010
---

Finding simple setup instructions for getting some OpenCV sample code up and running is a pain. They seem to make significant changes in each release, which means that an article providing setup instructions for an older version may not work for the latest. A lot of the instructions are geared towards being setup to recompile OpenCV, which you're probably not interested in when you're just getting started.

I'm currently using OpenCV 2.4.3, and working in VisualStudio 2010.

I found the following forum post to be very clear, and I was able to follow it to the letter successfully:

**[http://stackoverflow.com/questions/10901905/installing-opencv-2-4-in-visual-c-2010-express](http://stackoverflow.com/questions/10901905/installing-opencv-2-4-in-visual-c-2010-express)**

****A few additional notes:
****



	
  * If VisualStudio is open when you set your Path variable, you may need to relaunch VisualStudio. If the path variable has not been updated successfully, you’ll get an error when you try to _run_ an example (they'll compile fine) that it can’t find one of the DLLs on your computer.

	
  * The library names (the .lib files) that you reference in the project properties include the OpenCV version number (e.g., '243' for 2.4.3) and the letter ‘d’ for debug.


**Update:**

****I've recently updated to OpenCV 2.4.5 and confirmed that the steps in the above post still work perfectly.
