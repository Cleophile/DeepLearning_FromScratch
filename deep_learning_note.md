# Deep Learning
## Introduction
- We can think of each application of a different mathematical function as providing a new representation of the input
- Depth enables the computer to learn multistep computer program

## Feedforward network
- No recurrence of the signals
- Like a human neuron
- but are designed to achieve statistical generalization
- visible layer -> hidden layers -> output layer

Gradient decent algorithm for feedforward network:
Non-convex
Important initialize all weights to zero or small positive values

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

Advantage:
- Well defined loss function as long as distribution $P$ is defined  

Disadvantage:
- Can derive unlimited reward in some cases (Behaves like logistic)  



### Practice with TensorFlow
Details in `feedforward_keras.py`

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
- sparse interactions: sparse matrixed
- parameter sharing
- equivariant representations

### Pooling

A pooling function replaces the output of the net at a certain location with asummary statistic of the nearby outputs.  
Invariance to translation means that if wetranslate the input by a small amount, the values of most of the pooled outputsdo not change.  

