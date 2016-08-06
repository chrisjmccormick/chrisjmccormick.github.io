---
author: chrisjmccormick
comments: true
date: 2014-08-19 23:24:56+00:00
layout: post
link: https://chrisjmccormick.wordpress.com/?p=5983
published: false
slug: hidden-markov-models-tutorial-notes
title: Hidden Markov Models - Tutorial Notes
wordpress_id: 5983
categories:
- Tutorials
tags:
- Hidden Markov Models
---

I'm currently trying to gain an understanding of Hidden Markov Models, specifically as they're applied to speech recognition.

When learning about Gaussian Mixture Models, I began with a set of tutorial slides from Professor Andrew Moore at Carnegie Melon. Since I found his slides on GMM helpful, I thought I'd try working through his HMM slides as well.

Here is the link to the [tutorials section](http://www.cs.cmu.edu/~awm/tutorials.html) of his website,  and here is the link the to the [HMM slide set](http://www.autonlab.org/tutorials/hmm.html).

The intent of this post, then, is not to provide a standalone tutorial of HMMs, but simply to offer my notes on his tutorial.


### **State Transition Table**


(Slide 9)

This slide introduces this state transition probability table, with the following equation.

[![StateTransitionTable_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/statetransitiontable_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/statetransitiontable_eq.png)

This table records the probabilities for all of the possible state changes. It's just a matrix with one row per current state and one column per next state.


### **Probability of Arriving At A State**


(Slide 17)

We're now trying to answer the question "what's the probability of the next state being j if we don't know the current state?"

We're given the following equation for determining this.

[![NextStateCurrentUnknown_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/nextstatecurrentunknown_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/nextstatecurrentunknown_eq.png)

Note that we’re selecting a particular column of the matrix _a_, and summing over it, so the result is a single value for a given j.

We're assuming that the matrix a is given by the problem definition, but we _don't_ yet know p_t(i), except at t = 0. The solution is on the next slide.

(Slide 18 & 19)

We're going to create a table of all values p_t(j) for all t (one per row) and all j (one per column). To do this, we start at t = 0, because we know the state at t = 0, then we can compute the probabilities at t = 1, then at t = 2, and so on.

This is much more efficient than the "stupid way", and is an example of dynamic programming. Dynamic programming techniques just leverage the re-use of intermediate state information. That is, dynamic programming techniques are more efficient because they remove redundant calculations.

Once this table is calculated, we can easily calculate probabilities for questions like, "what is the probability of the robot crushing the human at step 5 (given the initial state at step 0, and assuming both are moving blindly/randomly)?"


### Summary of Common HMM Tasks


**Task #1: State Estimation**

(Slide 31)

We can use HMMs to try and determine our current state given the sequence of observations leading up to now.

This is called "State Estimation" and is represented by the following equation:

[![StateEstimationProblem_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/stateestimationproblem_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/stateestimationproblem_eq.png)

An example is given which appears to be the question of what is this person's current state (standing, on the PC, in class, etc.) based on the sequence of observations plotted. Note that the "observations" are not direct observations of the person's activity, but rather (I'm assuming) physiological measurements (heart rate, EKG?).


### **Task #2: Most Probable Path**


(Slide 34)

We can also use HMMs to determine the most probable path that lead us to our current state. That is, based on a set of observations over time, what do we think the real states were? The example here is determining when a person woke up, when they arrived at class, etc.

**Task #3: Learning HMMs from Data**

(Slide 37)

Another important task is learning HMMs from data. The idea is to learn a model from a training set which could then be applied to new data in order to answer questions 1 or 2 (i.e., What is the current state? Or what is the sequence of states that produced this set of observations?)


### Example Complete HMM


(Slides 41 - 48)

These slides walk through a simple example of generating a series of observations from a pre-defined HMM.

The way the observations work in this example is that we only observe one of the two letters in the actual state. So for each possible observation (X, Y, or Z) there are two possible actual states.


### State Estimation with Observations


(Slides 54 - 59)

Here we're look at, given a completely defined HMM, what is the most likely state given a sequence of observations.

What we want to arrive at is the probability of a given state given a sequence of observations.

[![ProbabilityOfStateWithObservations_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/probabilityofstatewithobservations_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/probabilityofstatewithobservations_eq.png)



In order to arrive at this, however, we first need to define a related value. We first want to calculate the probability of our HMM generating a particular sequence of observations _and_ arriving at a given final state.

[![ProbabilityOfAnObservationPath_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/probabilityofanobservationpath_eq.png?w=470)](https://chrisjmccormick.files.wordpress.com/2014/08/probabilityofanobservationpath_eq.png)

The caret-like operator is actually a set operator meaning "and".

The way we will calculate alpha efficiently is using a similar technique to what we looked at 17 - 19, where we looked at the probability of arriving at a given state (there, of course, we were doing it without considering any observations).

[![ProbabilityOfStateGivenObservations_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/probabilityofstategivenobservations_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/probabilityofstategivenobservations_eq.png)

Note the similarity to our earlier equation. Again, we can calculate this by first calculation alpha_1, then alpha_2, and so on. This is what he means in the notes by implementing this "recursively".


### Most Probable Path Given Observations


(Slides 60 - 66)

In this task, we want to figure out the most likely sequence of states which produced our set of observations. This is expressed with the following equation:

[![MostProbablePathObjective_Eq](http://chrisjmccormick.files.wordpress.com/2014/08/mostprobablepathobjective_eq.png)](https://chrisjmccormick.files.wordpress.com/2014/08/mostprobablepathobjective_eq.png)



'argmax' means find the value of Q which maximizes the probability. 'argmax' and 'max' both try to maximize the function, but 'max' returns the function value while 'argmax' returns the argument that produced that value.

A practical example of this "most probable path" problem is given on slide 66--speech recognition. If the audio signal is the sequence of observations, then what is the most likely sequence of "phones" (phones are "speech sounds") that produced these observations?

The Viterbi algorithm presented on slides 62 - 66 appears to follow the same iterative procedure as the others. We can directly calculate the values at t = 1, then use those to calculate the values at t = 2, and so on.


### Learning An HMM From Data


(Slides 67 - 82)

We'll be using Expectation-Maximization to learn the HMM from a set of observations.

This is an iterative technique, where you perform two steps over and over until convergence. In the "Expectation" step we will use our current estimation of the HMM and apply it to the data to generate some probabilities. Then, in the "Maximization" step, we'll re-calculate / update our HMM using the probabilities calculated in the first step. You go back and forth, applying the HMM then updating the HMM, until the model stops changing (you've converged).


