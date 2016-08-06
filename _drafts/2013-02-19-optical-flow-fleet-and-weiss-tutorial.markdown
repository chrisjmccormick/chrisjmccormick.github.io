---
author: chrisjmccormick
comments: true
date: 2013-02-19 16:46:33+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5420
published: false
slug: optical-flow-fleet-and-weiss-tutorial
title: Optical Flow - Fleet & Weiss Tutorial
wordpress_id: 5420
---

From what I've read so far, Optical Flow is more of a concept than a specific algorithm. It deals with tracking objects through a scene.

I've been reading through the following tutorial on the subject. This post contains my reading notes, which you may find helpful.

http://www.cs.toronto.edu/~fleet/research/Papers/flowChapter05.pdf


### Page 1


The following equation contained some notation that I had to look up.

[![notation](http://chrisjmccormick.files.wordpress.com/2013/02/notation.png)](http://chrisjmccormick.files.wordpress.com/2013/02/notation.png)





	
  * The arrow over the 'x' indicates that x(t) is a vector rather than a scalar.

	
  * The three bar equal sign means "equivalent to". In this case, I think it could be read as "can also be written as".

	
  * The superscript capital T means 'Transpose'. In this case, it just means that it should be written as a column vector (single column) instead of a row vector (single row). This notation is probably used just to avoid printing the vector as a column.







