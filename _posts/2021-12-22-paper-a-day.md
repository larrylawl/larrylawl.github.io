---
title: 'A Paper A Day'
date: 2022-12-22
permalink: /posts/2021/01/a-paper-a-day/
tags:
  - research
---
Just me trying to inculcate a habit of reading papers consistently...

## Motivation
As I step into this world of research, I realised I'm still in the midst of discovering what *I'm truly interested in*. To this end, I hope to read papers solely because I'm interested in it (instead of, say, to do literature review for my current research).

From a professional standpoint, I hope for this consistency pays off in terms of ideating my own novel ideas, inspired by the ideas from the community.

## Paper a day
Papers which I enjoyed are in **bold**.

### December
<!-- 3/31 -->
![10](https://progress-bar.dev/10)

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