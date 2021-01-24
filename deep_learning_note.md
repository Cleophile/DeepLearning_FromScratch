# Deep Learning
## Goal of Learning

![](./pic/171608730810_.pic.jpg)

## Introduction
- We can think of each application of a different mathematical function as providing a new representation of the input  
- Depth enables the computer to learn multistep computer program; later instructions can refer back to the results of earlier instructions  

## Feedforward network
- No recurrence of the signals  
- Like a human neuron  
- but are designed to achieve statistical generalization  
- visible layer -> hidden layers -> output layer  

Gradient decent algorithm for feedforward network:  
- Non-convex, thus what we get is usually a local minimum  
- Important initialize all weights to zero or small positive values  

### Loss Function
#### Maximum Likelihood 
Definition:
$$ J(\theta) = -\mathbb{E}_p (log(p(x|y))) $$

Integration form:

$$ J(\theta) = \int p(x) log(q(x)) dx $$

Cross Entropy: Same as the maximum likelihood
$$ -\sum_{i} p(x_i) log(q(x_i)) $$

Where $p(x)$ is the true probability, and $q(x)$ is the predicted probability

If one-hot is adapted, then $p(x)=1$ on the correct label and $p(x)=0$ otherwise. An element of cross entropy and be re-written as negative log:

$$ -log(p(x_{label}|y)) $$

Property:
- consistent estimator
- asymptotic normality

Advantage:
- Well defined loss function as long as distribution $P$ is defined  

Disadvantage:
- Can derive unlimited reward in some cases (Behaves like logistic)  

#### Why not square functions in classification:

Non-convex issues:

![](./pic/non_convex_square.jpg)

If applied not in classification but linear outputs, the log-likelihood is the same as minimizing the mean square:

$$ y \sim \frac{1}{\sqrt{2\pi}} e^{-\frac{(y-\hat{y})^2}{2 \sigma^2}} $$

Both in one-hot coding cross entropy and negative log-likelihood:

$$ -log(p(y)) = c + \frac{(y-\hat{y})^2}{2 \sigma^2} $$

### Functions of output units

$$ g(\vec{z}) = g(\mathbf{W}^T \vec{x} + \vec{b}) $$

The following functions all based on the same assumption, that the log of distribution is linear to its elements.

#### Sigmoid Function, for 0-1 results

$$ \sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x} $$

Recall SoftPlus Function: $\zeta(x)=log(1+e^x)$, which smoothens $x^+ = max\{0,x\}$
$$ J(\theta) = -log(\sigma(x)) = \zeta(-x) $$

This would amplify the gradient when x is of the wrong sign.  
However, if this is used in a mean-square error, from the graph of the sigmoid function, the gradient would become very small.  

#### Softmax Function, for multiple class

"soft": softmax function is continuous and diﬀerentiable

A generalized Sigmoid function:
$$ softmax(\vec{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

Sigmoid is the same as logistic regression, and the output can be viewed as multiple independent 0-1 classification missions.

Softmax is used when the multiple dimensions are mutual exclusive. $\sum w_i = 1$

$$ J(\theta) = -log(softmax(\vec{z})_i) = - z_i + log(\sum_j e^{z_j}) $$

#### Heteroscedastic gaussian and mixed gaussian

Re-define Gaussian:

$$ \sqrt{\frac{\beta}{2\pi}} exp(-\frac{1}{2} \beta (x-\mu)^2) $$

Heteroscedastic Gaussian: use $diag(\beta)$ as precision, which is an output value of $f(x,\theta)$

Mixed Gaussian:

$$p(y|x) = \sum_i p_i N(y,\mu_i(x),\Sigma_i(x)) $$

### Functions of hidden units

$$ g(\vec{z}) = g(\mathbf{W}^T \vec{x} + \vec{b}) $$

#### ReLU: Rectified Linear Units

$$ g(z) = max\{0, z\} $$

- **Excellent default** choice for hidden units  
- Preset $\vec{b}$ a small positive number to activate the gradient at the beginning  
- LeakyReLU: $g(z) = \alpha z$ for $z<0$  
- SoftPlus: $g(a)=log(1+e^a)$  

#### Maxout

$$ g(z)_i = max_{j \in G^{i}} z_j $$

This function cuts the original function into several pieces, and for each piece return the max value.  
Each group is of size $k$, $i=1,2,3,...,k$.  
For each group, suppose the input is m-size $x$, $\mathbf{W_i}$ is of size $m \times k$:
$$ g(z)_i = max \{ \mathbf{W_i}^T \vec{x} + \vec{b} \} = max(W_{i1}^T \vec{x} + b_1, W_{i2}^T \vec{x} + b_2,..., W_{ik}^T \vec{x} + b_k) $$

This function can be interpreted as a sectioned function: the function changes when the section of $x$ changes. When the number of sections $k$ is large enough, the function is able to simulate whatever functions.  

More normalizing for each layer

The number of parameters has noticably increased, however the excess parameters can prevent catastrophic forgetting.  

#### Sigmoid

With log-likelihood loss function, sigmoid function saturate when $z$ becomes large. A close-to-zero gradient makes learning impossible to proceed.  

#### tanh

$$ tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

Hard-tanh:

$$ g(a) = max\{-1,min\{1, a\}\} $$

#### RBF

$$ h_i = exp(-\frac{1}{\sigma_i^2} \|\mathbf{W}_{i} - \vec{x} \|_2^2) $$

The function is only active when $x$ is close to the "template" $W$

#### Linear

Consider $W = UV$

$W$ consists of $n\times p$ parameters, $U$ consists of $n \times q$ parameters, $V$ consists of $q \times p$

When $q$ is small, $(n+p)q$ can be smaller then $np$

### Practice with TensorFlow
Details in `feedforward_keras.py`

### Universal approximation theorem

A feedforward network with a linear output layer and at least one hidden layer with any "squashing" activation function (such as the logistic sigmoid activation function) can approximate any Borel measurablefunction from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.

`feedforward_keras.py`:
- 设计网络类
- 选择损失函数和优化器
- 设计训练流程和测试流程
- 循环训练和测试

## Optimization of Neural Networks

### 0-1 loss and log-p

Even when the 0-1 loss is zero, like when the possibility of the correct element is already the largest, there are still rooms for improvement.  
Log-p and cross-entropy:



### Stopping Criteria

Stops when loss function the validation set has stopped improving for several consecutive rounds.  
Notice that not necessarily when the gradient turns zero.

### Batch and MiniBatch

Used in the random gradient descent: use a small portion of the dataset and


## CNN
Convolutional networks are simply neural networks that use convolution in place of general matrixmultiplication in at least one of their layers  

### Convolution
Motivation of convolution: weighted average that gives more weight to recent measurements  
- input
- kernel

CNN convolution layer:
- Sparse interactions: sparse matrixed
- Parameter sharing
- Equivariant representations

### Pooling

A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs.  
Invariance to translation means that if wetranslate the input by a small amount, the values of most of the pooled outputsdo not change.  

