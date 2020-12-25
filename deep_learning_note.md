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

积分形式

Cross Entropy: Same as the maximum likelihood
$$ -\sum_{i} p(x) log(q(x)) $$

Where $p(x)$ is the true probability, and $q(x)$ is the predicted probability

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
