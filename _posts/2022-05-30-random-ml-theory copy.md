---
title: 'Random Machine Learning Theory'
date: 2022-05-30
permalink: /posts/2022/05/random-ml-theory/
tags:
  - research
---

## Machine Learning (ML)

<!-- PCA -->
**[Principal Component Analysis](https://www.youtube.com/watch?v=FgakZw6K1QQ)** 

**Summary**. The principal components of a collection of points in a real coordinate space are a sequence of $p$ unit vectors, where the $i$-th vector is the direction of a line that best fits the data while being orthogonal to the first $i-1$ vectors. Thus, the first vector is the line that best fits the data, while the second vector is the second best line while being orthotogonal to the first, etc.

**Motivation.** PCA is often used for dimensionality reduction. We can reduce from $n$ dimensions to $k$ by selecting only the first $k$ principal components.

**Algorithm.** 
Calculate mean of all values. Shift data such that mean is at the origin
![center and shift](/images/PCA/center-and-shift.png)

Find the first principal component (PC1). That is, start by drawing a random line through the origin. Slowly rotate such that it either 1) minimizes distances from the data to the line or 2) maximises the distances from the projected points to the origin.
![min or max](/images/PCA/min-or-max.png)

Next, find PC2, the next best fitting line given that it goes through the origin and is perpendicular to PC2. Repeat steps 1 and 2 until $k$. 

Suppose $k=2$. Project samples onto PC1 and PC2. Compute PCA plot based on projected samples.
![reconstruction](/images/PCA/reconstruction.png)

**How to evaluate each principal components?** Compute the sum of squared distances (SS) for the principal component. From point 2 of the algo, the larger this distance, the better. Thereafter, compute the variation and plot the scree plot. From the example below, limiting to two dimensions is good enough.
![scree plots](/images/PCA/scree-plots.png)

> Thanks Statquest for the explanation! Link [here](https://www.youtube.com/watch?v=FgakZw6K1QQ).

**[Kullback–Leibler divergence (KL divergence)](https://m.youtube.com/watch?v=SxGYPqCgJWM)**.
The KL divergence, $D_{KL}(P \Vert Q)$, is a numerical measure of the distance between two probability distributions. For example, consider the following distributions:

$$
X \sim Ber(p), \quad Y \sim Ber(q)
$$

We are interested in the distance between the distributions of $X$ and $Y$. Suppose $p=0.5, q=0.1$, we can tell quite obviously that $Y$ is different from $X$ by simply sampling from $X$. But how do we quantify this distance?

We can gauge this distance by computing the likelihood ratio:

$$
\frac{lik(p)}{lik(q)}
$$

If the distributions are similar, then the ratio should equal 1. The KL divergence builds on via:

$$
\frac{1}{n} log(\frac{lik(p)}{lik(q)}) = \frac{1}{n} log(\frac{p_1^{N_H} p_2^{N_T}}{q_1^{N_H} q_2^{N_T}})  
\\ = \frac{1}{n} (N_H logp_1 + N_T log p_2 - N_H log q_1 - N_T log q_2)
\\ = \frac{N_H}{n} logp_1 + \frac{N_T}{n} logp_2 - \frac{N_H}{n} log q_1 - \frac{N_T}{n} log q_2
\\ \approx p_1 logp_1 + p_2 logp_2 - p_1 log q_1 - p_2 log q_2 \quad (n \rightarrow \infty)
\\ = p_1 log \frac{p_1}{q_1} + p_2 log \frac{p_2}{q_2}
$$

This now follows the more general form (which you can refer to wiki for):

$$
D_{KL}(P \Vert Q) =  \sum_{x \in X} P(x) log(\frac{P(x)}{Q(x)})
$$

Some properties:
1. $D_{KL}(P \Vert Q) \geq 0$
2. $D_{KL}(P \Vert Q)  = 0$ if $P = Q$ almost everywhere.

> Thanks to this awesome youtube channel for the explanation! Link [here](https://m.youtube.com/watch?v=SxGYPqCgJWM).

## NLP

**[Word2Vec](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)** The motivation is to turn each to vectors. There are two primary algorithms to do so: CBOW and Skip-Gram. CBOW tries to predict a center word from the surrounding context. For each word, we learn two vectors: 1) v (input vector), when the word is in the context and 2) u (output vector), when the word is in the center. Skip-Gram tried to predict the surrounding word from the given center word. 

Training Word2Vec algorithms are expensive due to the large vocabulary size. Specifically, optimising the cost function and the softmax operator both involves looping through the vocabulary. We use negative sampling to circumvent the first problem and hierarchical softmax for the second.

## Math

**[Taylor Series](https://www.youtube.com/watch?v=3d6DsjIBzJ4)**(3Blue1Brown)

**Motivation.** To approximate non-polynomial function as a polynomial. Polynomials are in general easier to deal with (easier to compute, obtain derivatives etc).

Concretely, consider the problem of approximating $cos(x)$ as a polynomial $P(x) = c_0 + c_1 x + c_2 x^2$. We can approximate them by ensuring their derivatives at $x=0$ are the same.

$$
cos(0) = 1 \\
\frac{d cos(0)}{dx} = -sin(0) = 0 \\
\frac{d^2 cos(0)}{dx^2} = -cos(0) = -1 
$$

To ensure their derviatives at $x=0$ are the same,


$$
P(0) = c_0 \implies c_0 = 1 \\
\frac{dP}{dx} (0) = c_1 + 2c_2 (0) = c_1 \implies c_1 = 0 \\ 
\frac{d^2 P}{dx^2} (0) = 2c_2 \implies c_2 = -\frac{1}{2}
$$

Thus, $cos(x) \approx 1 - \frac{1}{2} x^2$. Notice how every derivative order is independent of each other and provides a unique information (i.e. $P(0)$ provides the intersection point, $P'(0)$ provides the correct gradient, $P''(0)$ provides the correct rate etc.)

![Taylor expansion examples](/images/papers/taylor-expansion-eg.png)