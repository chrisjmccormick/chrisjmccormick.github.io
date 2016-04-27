---
author: chrisjmccormick
comments: true
date: 2015-09-16 23:16:41 -0800
layout: post
link: https://chrisjmccormick.wordpress.com/2015/09/16/notes-on-pagerank/
slug: notes-on-pagerank
title: Notes on PageRank
wordpress_id: 6068
tags:
- Data Mining
- MATLAB
- PageRank
- Power Iteration
- Python
- Web Graph
---

This post is just intended to capture my notes on the PageRank algorithm as described in the Mining Massive Datasets course on Coursera.

It is described in detail in [chapter 5 of their free textbook](http://infolab.stanford.edu/~ullman/mmds/ch5.pdf), and you may also be able to access the [video lectures here](https://class.coursera.org/mmds-002/lecture) (PageRank is discussed in week 1).


## **History**


Because PageRank is used by Google for ranking search results, you'd assume it's name is derived from "Ranking Webpages"; however, it's actually named after one of the authors, Larry Page of Google fame. I think part of the reason that this distinction makes sense is that PageRank is useful for analyzing directed graphs in general, not just the web graph.


## Node Scoring


The scores of the nodes on the graph are all inter-related as follows.



	
  * Each node distributes its score to each of the nodes it points to. If it points to 3 nodes, it contributes 1/3 of its score to each of them.

	
  * A node's score is given by the sum of the contributions from all of the nodes pointing to it.


This can be expressed with linear algebra--we'll have a matrix 'M' that represents the graph connections, and a column vector 'r' that holds the score for each node.

M is an adjacency matrix specifying the connections between the nodes. Instead of putting a '1' where the connections are, however, we put a fraction: 1 / the number of nodes it points to.

[![SimpleGraph](https://chrisjmccormick.files.wordpress.com/2015/09/simplegraph.png)](https://chrisjmccormick.files.wordpress.com/2015/09/simplegraph.png)[![SimpleGraph_M](https://chrisjmccormick.files.wordpress.com/2015/09/simplegraph_m.png)](https://chrisjmccormick.files.wordpress.com/2015/09/simplegraph_m.png)

We can capture our description of the relationship between the scores of the nodes using the matrix 'M' and 'r' as the expression

[![Eq_r_Mr](https://chrisjmccormick.files.wordpress.com/2015/09/eq_r_mr.png)](https://chrisjmccormick.files.wordpress.com/2015/09/eq_r_mr.png)


## Solving for the Scores


Our equation for the node scores has the same form as the equation for an eigenvector, 'x':

[![Eq_Egeinvector](https://chrisjmccormick.files.wordpress.com/2015/09/eq_egeinvector.png)](https://chrisjmccormick.files.wordpress.com/2015/09/eq_egeinvector.png)

This means that we can interpret our score vector 'r' as an eigenvector of the matrix M with eigenvalue lambda = 1.

Because of some properties of M (that hold because of how we defined it), our score vector 'r' will always be the first eigenvector of M, and will always have an eigenvalue of 1.

One way to calculate 'r', then, would just be to run eigs on the matrix M and take the principal eigenvector.

In Matlab:

    
    % Adjacency matrix for the graph
    M = [0.5, 0.5, 0;
         0.5,   0, 1;
           0, 0.5, 0]
    
    % 'eigs' will return the eigenvectors in order of the magnitude
    % of the eigenvalue.
    [V, D] = eigs(M);
    
    % The PageRank scores for the nodes will be in the first eigenvector.
    r = V(:, 1)
    
    % We can also verify that M*r gives back r...
    fprintf('M * r =\n');
    M * r




In Python:

    
    M = [[0.5, 0.5, 0],
         [0.5,   0, 1],
         [  0, 0.5, 0]]
     
    # Find the eigenvectors of M. 
    w, v = numpy.linalg.eig(M)




## Power Iteration


I used the 'eig' functions above just to show that you can. Because we are only interested in the principal eigenvector, however, there's actually a computationally simpler way of finding it. It's called the power iteration. You just initialize the score vector 'r' by setting all the values to 1 / the number of nodes in the graph. Then repeatedly evaluate r = M*r. Eventually, 'r' will stop changing and converge on the correct scores!

    
    % Adjacency matrix for the graph
    M = [0.5, 0.5, 0;
         0.5,   0, 1;
           0, 0.5, 0]
    
    % Initialize the scores to 1 / the number of nodes.
    r = ones(3, 1) * 1/3;
    
    % Use the Power Iteration method to find the principal eigenvector.
    iterNum = 1;
    
    while (true)
        % Store the previous values of r.
        rp = r;
     
        % Calculate the next iteration of r.
        r = M * r;
     
        fprintf('Iteration %d, r =\n', iterNum);
        r
     
        % Break when r stops changing.
        if ~any(abs(rp - r) > 0.00001)
            break
        end
     
        iterNum = iterNum + 1;
    end
    
    % We can also verify that M*r gives back r...
    fprintf('M * r =\n');
    M * r


Note: If you run this power iteration method, you'll find that it gives a different score vector than the 'eigs' approach. However, both score vectors appear to be correct. Honestly, I'm not sure why this is the case--perhaps this simple exmaple matrix 'M' is reducible, and therefore there is not just a single unique solution?


## Fixing the Web Graph


There is a problem with how we've defined PageRank so far. It only works if the matrix M is stochastic (the values in a column sum to 1) and aperiodic (it doesn't contain any cycles). Our simple example satisfied both of these, but the web graph does not. For example, if you have a webpage with no outlinks, then it's column in M will be all zeros, and M is no longer stochastic. Or, if you have two pages which point to each other, but no one else, then you have a cycle, and M is no longer aperiodic.

We fix this by tweaking the web graph. We add a link between each page and every other page on the internet, and just give these links a very small weight.

Here's how we express this tweak algebraically. 'M' is our original matrix, and A is our new modified matrix that we will use to determine the scores.

[![Eq_FinalFormulation](https://chrisjmccormick.files.wordpress.com/2015/09/eq_finalformulation.png)](https://chrisjmccormick.files.wordpress.com/2015/09/eq_finalformulation.png)

And here's the Matlab code to run PageRank with this modified graph. If you run this code, you'll find that it produces close to the same result as before, but the implementation is now robust against dead ends and cycles in the graph.

    
    % Adjacency matrix for the graph
    M = [0.5, 0.5, 0;
         0.5,   0, 1;
           0, 0.5, 0]
    % Tweak the graph to add weak links between all nodes. 
    beta = 0.85;
    n = size(M, 1);
    
    A = beta * M + ((1 - beta) * (1 / n) * ones(n, n))
    % Initialize the scores to 1 / the number of nodes.
    r = ones(3, 1) * 1/3;
    
    % Use the Power Iteration method to find the principal eigenvector.
    iterNum = 1;
    
    while (true)
        % Store the previous values of r.
        rp = r;
     
        % Calculate the next iteration of r.
        r = M * r;
     
        fprintf('Iteration %d, r =\n', iterNum);
        r
     
        % Break when r stops changing.
        if ~any(abs(rp - r) > 0.00001)
            break
        end
     
        iterNum = iterNum + 1;
    end
    
    % We can also verify that M*r gives back r...
    fprintf('M * r =\n');
    M * r
    
    
    
    
