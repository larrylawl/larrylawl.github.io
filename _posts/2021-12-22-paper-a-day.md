---
title: 'A Paper A Day'
date: 2021-12-22
permalink: /posts/2021/01/a-paper-a-day/
tags:
  - research
---
Just me trying to inculcate a habit of reading papers consistently...

## Motivation
As I step into this world of research, I realised I'm still in the midst of discovering what *I'm truly interested in*. To this end, I hope to read papers solely because I'm interested in it (instead of, say, to do literature review for my current research).

From a professional standpoint, I hope for this consistency pays off in terms of ideating my own novel ideas, inspired by the ideas from the community.

<!-- 1. Love math, 2. like problem domain (e.g. NLP, CV) - very exciting to try learn solutions -->
<!-- Ethical/Environmental NLP? -->

## Paper a day
Papers which I enjoyed are in **bold** or prepended with [F].

### May
> Back after ending my semester!

[Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://aclanthology.org/P19-1041.pdf) (Vineet John, Lili Mou, Hareesh Bahuleyan, Olga Vechtomova, ACL2019)
- In this paper, they addressed the problem of **disentangling** the latent space of neural networks for text generation. Their model is built on an **autoencoder** that encodes a sentence to the latent space (vector representation) by learning to reconstruct the sentence itself. They would like the latent space to be disentangled with respect to different features, namely, style and content in their task.
> **Disentanglement** is the problem of disentagling text into its content and attribute in the latent space, and apply generative modelling.
- They design a systematic set of auxiliary losses, enforcing the separation of style and content latent spaces. 
  - In particular, the multi-task loss operates on a latent space to ensure that the space does contain the information we wish to encode. 
  - The adversarial loss, on the contrary, minimizes the predictability of information that should not be contained in a given latent space. 
> Example with style-oriented loss. For multi-task loss for style, they build a softmax classifier on the style space to predict the style label. In the paper, the style space is sentiment classification, thus the style label is positive or negative. For adversarial loss for style, they train a separate classifier, called an adver- sary, that deliberately discriminates the style label based on the content vector c. Then, the encoder is trained to encode a content space from which its adversary cannot predict the style.

- Earlier works only consider style space and ignore the content space, as it is hard to formalize what "content" actually refers to. In their work, they propose to approximate content information by bag-of-words (BoW) features, where they focused on style-neutral, non-stop words.

[Deep Learning for Text Style Transfer: A Survey](https://arxiv.org/pdf/2011.00416.pdf) (Di Jin, Zhijing Jin, Zhiting Hu, Olga Vechtomova, Rada Mihalcea, Computational Linguistics Journal 2022)

**Goal.** The goal of text style transfer (TST) is to automatically control the style attributes of text while preserving the content. Formally, we dnote the target utterance as $\mathbf{x'}$ and the target discourse style attribute as $a'$. TST aims to model $p(\mathbf{x}'|a, \mathbf{x})$, where $\mathbf{x}$ is a given text carrying a source attribute value $a$. For example, 
- Source sentence $\mathbf{x}$: *"Come and sit!"*. Source attribute *a*: *Informal*.
- Target sentence $\mathbf{x'}$: *"Please consider taking a seat"*. Target attribute *a'*: *Formal*.

Ideally, parallel data is available, which allows us to apply standard seq2seq models directly. However, parallel data is often not available.

**Methods (on non-parallel data)**
- **Disentanglement.** Disentagles text into its content and attribute in the latent space, and apply generative modelling.
- **Prototype Editing.** 
  - Step 1: Detect attribute markeres of *a* in the input sentence $\mathbf{x}$, and delete them, resulting in a content-only sentence.
  - Step 2: Retrieve candidate attribute markers carrying the desired attribute $a'$.
  - Step 3: Infill the sentence by adding new attribute markers and make sure the generated sentence is fluent. 
- **Pseudo Data Construction.** Construct pseudo-parallel data and train the model as if in a supervised way.

![TST survey overview](/images/papers/tst-survey.png)

### January
<!-- 6/31 -->
![24](https://progress-bar.dev/24)

> I'll be taking a break with semester resuming! Hoping to finish my schooling years on a high 😎

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


**[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848.pdf)**(Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le, NeurIPS 2020)
In a nutshell, consistency training methods simply regularize model predictions to be invariant to small noise applied to either input examples or hidden states. This framework makes sense intuitively because a good model should be robust to any small change in an input example or hidden states. This work observes that advanced data augmentation methods also work well with consistency training methods. They hence propose to substitute the traditional noise injection methods with high quality data augmentation methods in order to improve consistency training.

**[Controllable Neural Dialogue Summarization with Personal Named Entity Planning](https://arxiv.org/pdf/2109.13070.pdf)**
In dialogue summaries, perspective is important. For example, John and Tony's perspective prioritises different summaries from the same source text. 

![Dialogue summary examples](/images/papers/controllable-dialogue-summarisation.png)

However, a neural model with a general summarizing purpose may overlook information that is important to a specific person's perspective. This paper introduces a controllable dialogue summarisation framework, in which module the generation process with personal named entity plannings. More specifically, as shown in the figure above, a set of personal named entities1 (in color) are extracted from the source dialogue, and used in a generation model as a conditional signal.

<!-- ELBO distance -->
**[Coreference-Aware Dialogue Summarization](https://arxiv.org/abs/2106.08556)** (Zhengyuan Liu, Ke Shi, Nancy Chen. Coreference-Aware Dialogue Summarization, SIGDIAL 2021 (Best Paper Award)) This paper tackles the problem of text summarisation. Recent work focuses on single speaker content such as news articles but not dialogues. Dialogue summarisation is difficult because of 1) multiple speakers, 2) speaker role shifting and 3) ubiquitous referring expressions (e.g. speakers referring to themselves and each other). These challenges justify the usefulness of leveraging coreference information: when two or more expressions in a text refer to the same thing (e.g. Bill said he would come; the proper noun Bill and pronoun he refer to the same person). Large scale pretrained model are shown to implicitly model lower level linguistic knowledge such as PoS and syntactic structure. Since entities are linked to each other in coreference chains, this paper propose to add a graph neural layer that could characterize underlying coreference information structure, which can improve dialogue summarisation performance.



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
<!-- Glove -->

**[Word2Vec](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)** The motivation is to turn each to vectors. There are two primary algorithms to do so: CBOW and Skip-Gram. CBOW tries to predict a center word from the surrounding context. For each word, we learn two vectors: 1) v (input vector), when the word is in the context and 2) u (output vector), when the word is in the center. Skip-Gram tried to predict the surrounding word from the given center word. 

Training Word2Vec algorithms are expensive due to the large vocabulary size. Specifically, optimising the cost function and the softmax operator both involves looping through the vocabulary. We use negative sampling to circumvent the first problem and hierarchical softmax for the second.

### December
<!-- 8/31 -->
![25](https://progress-bar.dev/25)

**[Vocabulary Learning via Optimal Transport for Neural Machine Translation](https://arxiv.org/pdf/2012.15671.pdf)**(ACL2021, Best Paper award, Jingjing Xu, Hao Zhou, Chun Gan, Zaixiang Zheng, Lei Li) This paper considers the problem of vocabulary construction for neural machine translation and other NLP tasks. Current approaches primarily only consider frequency (entropy) but not size (e.g. BPE, Wordpiece tokenisation). However, there exist a tradeoff between the vocabulary entropy and the size: increase in vocabulary size decreases corpus entropy, but too many tokens cause token sparsity, which hurts model learning. Given this trade-off, this paper proposes to use the concept of *Marginal Utility* to balance it: in economics, marginal utility is used to balance the benefit and the cost and we use MUV to balance the entropy (benefit) and vocabulary size (cost). To maximize the marginal utility in tractable time complexity, they reformulated the optimisation objecitve into an optimal transport problem. Intuitively, the vocabularization process can be regarded as finding the optimal transport matrix from the character distribution to the vocabulary token distribution.

[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) Graph neural networks are commonly used for data which are naturally represented as graphs (.e.g molecules, social networks, citation networks). Tasks include: graph-level task (e.g. what the molecule smells like), node level task, edge-level task (image scene understanding). Solving this tasks will involve learning embeddings for graph attributes (e.g. nodes, edges, global). To make these learned embeddings aware of graph connectivity, one way is via message passing. Message passing works in three steps:
1. For each node in the graph, *gather* all the neighboring node embeddings (or messages).
2. Aggregate all messages via an aggregate function (like sum).
3. All pooled messages are passed through an update function, usually a learned neural network


[Using the Output Embedding to Improve Language Models](https://aclanthology.org/E17-2025.pdf)(EACL2017, Ofir Press, Lior Wolf) This paper showed that the topmost weight matrix of the neural language models (i.e. layer right before the softmax) constitutes a valid word embedding layer. As such, we can tie both layers together - known as *weight tying* - so as to reduce the number of parameters.

[Using “Annotator Rationales” to Improve Machine Learning for Text Categorization](https://aclanthology.org/N07-1033/) (NAACL 2007, Omar Zaidan, Jason Eisner, Christine Piatko) This paper proposes SVM training that can incorporate annotator rationales: subset of the input text that justify the task classification. Specifically, the paper constructs from positive examples $x_i$ positive "not-quite-as-positive" examples $v_{ij}$, which are obtained by masking out rationale substrings. In addition to the usual SVM constraint on positive examples that $w \cdot x_i \geq 1$, we also want (for each j) that $w \cdot x_i - w \cdot v_{ij} \geq \mu$, where $\mu \geq 0$ controls the size of the desired margin between original and contrast examples; that is, the margin of the original examples should ideally be much larger than the contrast examples.

[Rationale-Augmented Convolutional Neural Networks for Text Classification](https://arxiv.org/pdf/1605.04469.pdf) (ACL2016, Ye Zhang, Iain Marshall, Byron C. Wallace) This paper augments CNNs with rationales for the task of text classifications. Specifically, it uses CNNs for sentence modelling. Thereafter, an estimator is trained to produce the probability if the given sentence is 1) positive rationale (when a rationale sentence appears in a positive document), 2) negative rationale, 3) neutral rationale (non-rationale sentences). Thereafter, a document level classifier makes a task classification based on the weighted sum of the sentence vectors, with the weights specified by the probability of the sentence being a rationale or not.

> The authors also experimented with having only two sentence classes: rationales and non-rationales, but this did not perform as well as explicitly main- taining separate classes for rationales of different polarities.

**[Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks](https://arxiv.org/pdf/1906.04393.pdf)** (ACL2019, Yi Tay, Aston Zhang, Luu Anh Tuan, Jinfeng Rao, Shuai Zhang, Shuohang Wang, Jie Fu, Siu Cheung Hui) This paper explores computation in the Quarternion space (i.e. hypercomplex numbers) as an inductive bias. Specifically, quaternions $Q$ comprise of a real and three imaginary components in which interdependencies are naturally encoded during training (e.g. RGB scenes or 3D human poses) via the Hamilton product. 

$$ Q = r + x\mathbf{i} + y\mathbf{j} + z\mathbf{k} $$

Hamilton products $\otimes$, which represents the multiplication of two quaternions $W$ and $Q$, have fewer degrees of freedom, enabling up to four times compression of model size. Specifically, $W \otimes Q$ can be expressed as:

$$
\left[\begin{array}{cccc}
W_{r} & -W_{x} & -W_{y} & -W_{z} \\
W_{x} & W_{r} & -W_{z} & W_{y} \\
W_{y} & W_{z} & W_{r} & -W_{x} \\
W_{z} & -W_{y} & W_{x} & W_{r}
\end{array}\right]\left[\begin{array}{l}
r \\
x \\
y \\
z
\end{array}\right]
$$

Note that there are only 4 distinct parameter variable elements (4 degrees of freedom) in the weight matrix. In the real-space feed-forward, there will instead be 16 different parameter variables. In summary, Quaternion neural models model interactions between components via using hypercomplex numbers and provides efficiency via the Hamilton product, which have fewer degrees of freedom in comparison to matrix products in the real space.

<!--  -->
<!-- hypercomplex numbers - model interactions -->
<!-- hamilton product have lesser degrees of freedom -->

[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/pdf/1810.09536.pdf)(ICLR 2019, Best Paper, Yikang Shen, Shawn Tan, Alessandro Sordoni, Aaron Courville) While the standard LSTM architecture allows different neurons to track information at different time scales, it does not have an explicit bias towards modeling a hierarchy of constituents. This paper proposes to add such an inductive bias by ordering the neurons; a vector of master input and forget gates ensures that when a given neuron is updated, all the neurons that follow it in the ordering are also updated. 

[When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute](https://arxiv.org/abs/2102.12459) (EMNLP 2021 Outstanding Paper, Tao Lei) This paper combines attention with recurrent neural network that achieves strong computational efficiency; the simple recurrent unit is efficient because 1) it combines the three matrix multiplications across all time steps as a single multiplication and 2) implements element wise operations as a single CUDA kernel to accelerate computation.

### Archive
[Few-shot Conformal Prediction with Auxiliary Tasks](https://arxiv.org/abs/2102.08898) (ICML 2021, Adam Fisch, Tal Schuster, Tommi Jaakkola, Regina Barzilay) To circumvent the problem of *prediction sets* becoming too large when limited data is available for training, this paper casts *conformal prediction* as a meta-learning paradigm over exchangeable collections of auxiliary tasks.

[Rethinking Positional Encoding in Language Pre-training](https://openreview.net/pdf?id=09-528y2Fgf) (Guolin Ke, Di He, Tie-Yan Liu) This paper 1) unties the correlations between positional and word embeddings by computing them separately with different parameterizations before adding them (as opposed to adding before computing) and 2) unties [CLS] symbol from other positions, making it easier to capture information from all positions (as opposed to being biased to the first several words if we apply relative positional encoding)