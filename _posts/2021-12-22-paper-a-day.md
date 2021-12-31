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
Papers which I enjoyed are in **bold**.

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