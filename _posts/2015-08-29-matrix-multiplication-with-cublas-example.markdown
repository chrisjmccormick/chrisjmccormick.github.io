---
author: chrisjmccormick
comments: true
date: 2015-08-29 00:42:50 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2015/08/29/matrix-multiplication-with-cublas-example/
slug: matrix-multiplication-with-cublas-example
title: Matrix Multiplication with cuBLAS Example
wordpress_id: 6064
tags:
- cuBLAS
- CUDA
- Example Code
- Matrix Multiplication
- matrixMulCUBLAS
- NVIDIA
---

This post provides some overview and explanation of NVIDIA's provided sample project 'matrixMulCUBLAS' for super-fast matrix multiplication with cuBLAS. The example can be a little confusing, and I think it warrants some explanation.

In my installation, this sample can be found here:

C:\ProgramData\NVIDIA Corporation\CUDA Samples\v5.5\0_Simple\matrixMulCUBLAS\


## What does it do?


This example generates two matrices, A and B, filled with random values. The matrices are single precision floating point. The example is going to calculate C = A * B, and it times how quickly CUDA can do this (measured as gigaflops).

Using the default parameters, this example calculates (with matrix sizes shown as [rows x columns]):


C  [640 x 320]  =  A  [640 x 320]  *  B  [320 x 320]


A frustrating source of confusion in this example is that B is labeled / generated as having 640 rows, but only the first 320 rows are actually used in the matrix multiplication operation; more on that later. Note, though, that the results and performance measurements are still correct despite this oversight--the example isn't technically "broken".

The example also includes a naive, double-for-loop C/C++ implementation of matrix multiplication on the CPU. The results of the two matrix multiplications are compared to ensure that the CUDA implementation is giving the right answer.


## Matrix Sizes & Data


There are two sources of confusion with this example. One is a legitimately important detail of working with CUDA that you need to consider and that is worth learning. The other is just stupid and frustrating, and hopefully NVIDIA will fix it in a future version of the example, even though it doesn't strictly break the code.

In summary:



	
  1. The valid point: C/C++ assumes matrices are in row-major order, but CUDA assumes they are in column major order. To get the right result _without doing any explicit transpose operations_, you can switch the order of the matrices when calling 'gemm'.

	
  2. The frustrating point: Matrix B is allocated as being 640 rows by 320 columns, but only the first 320 rows are actually used in the calculation!


So what's this business about row-major and column-major order? It has to do with how matrices are actually laid out in memory. Row-major order means that all of the values in a row are contiguous in memory. Check out this nice example I stole from [Wikipedia](https://en.wikipedia.org/wiki/Row-major_order#Explanation_and_example):

This matrix



    ![ \begin{bmatrix}
11 & 12 & 13 \\
21 & 22 & 23 \end{bmatrix}](https://upload.wikimedia.org/math/9/0/7/907659c6f9ccbf7a29f7e13d44560b5c.png)
Would be stored as follows in the two orders:
<table class="multicol" >
<tbody >
<tr >

<td >
<table class="wikitable" >Column-major order (CUDA)
<tbody >
<tr >
Address
Value
</tr>
<tr >
0

<td >11
</td>
</tr>
<tr >
1

<td >21
</td>
</tr>
<tr >
2

<td >12
</td>
</tr>
<tr >
3

<td >22
</td>
</tr>
<tr >
4

<td >13
</td>
</tr>
<tr >
5

<td >23
</td>
</tr>
</tbody>
</table>

</td>

<td >
<table class="wikitable" >Row-major order
(C / C++)
<tbody >
<tr >
Address
Value
</tr>
<tr >
0

<td >11
</td>
</tr>
<tr >
1

<td >12
</td>
</tr>
<tr >
2

<td >13
</td>
</tr>
<tr >
3

<td >21
</td>
</tr>
<tr >
4

<td >22
</td>
</tr>
<tr >
5

<td >23
</td>
</tr>
</tbody>
</table>

</td>
</tr>
</tbody>
</table>
When a matrix is passed to CUDA, the memory layout stays the same, but now CUDA assumes that the matrix is laid out in column-major order. This won't cause a buffer overrun, but what it does is effectively transpose the matrix, without actually moving any of the data around in memory.

The assumption in NVIDIA's example is that, as the user, you want to calculate C = A * B. Your matrices are in C++, so they're in row-major order, and you want your result matrix C to similarly be in row-major order as well. If you pass the matrices in reverse order, CUDA will calculate B' * A', which is equal to C'. But when you take the result into C++, there's the implicit transpose again, so what you actually get is C.

Here's how you interpret the parameters in the code.

The variables uiWA, uiHA, uiWB, uiHB, uiWC, and uiHC are all from the perspective of the row-major C++ matrices. So uiWA is the width (number of columns) in A, uiHA is the height (number of rows) in A, etc.

The default values are as follows

uiWA, uiWB, uiWC = 320

uiHA, uiHB, uiHC = 640

But remember the second point about only using half of B? To make this example more sensical, the default for uiHB should really be 320, since that's all that's actually used of B. One piece of evidence to confirm this--if you look at the actual 'gemm' call, you'll notice that the uiHB parameter is unused. Instead, that dimension of the matrix is inferred as being equal to uiAW, which is 320. Want even further proof? Change uiHB to 320 (matrix_size.uiHB = 2 * block_size * iSizeMultiple;) and the code will still run, and the results validation will still pass.

So what we're going to calculate in this example is C  [640 x 320]  =  A  [640 x 320]  *  B  [320 x 320]

Now let's make sense of the parameters in the 'gemm' call. The parameters are messy because we've defined them with respect to the row-major matrices, but CUDA wants to know the parameters assuming that the matrices are in column-major order.

'gemm' asks for three matrix dimensions (here's a link to the [API doc](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)):



	
  * 'm' - "number of rows of matrix op(A) and C."  -- Our first operand is B', so the number of rows is in the first operand is uiBW

	
  * 'n' - "number of columns of matrix op(B) and C." -- Our second operand is A', so the number of columns in the second operand is uiAH

	
  * 'k' - "number of columns of op(A) and rows of op(B)." -- B' has uiBH columns, and A' has uiAW rows.

	
    * This is where the example's flawed. If you passed uiBH here, it wouldn't work!








## Timing Details


The example also measures the gigaflops that you're getting from your GPU. Some important notes:



	
  * It does not include the time to copy the generated data to the GPU.

	
  * It repeats the matrix multiplication 30 times, and averages the time over these 30 runs.


For the matrix multiplication operation:

C [m x n] = A [m x k] * B [k * n]

The number of floating point operations required is 2 * m * k * n. The factor of two is there because you do a multiply and an accumulate for each pair of values in the calculation.


