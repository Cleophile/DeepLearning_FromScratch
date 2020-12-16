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
$$ J(\theta) = -\mathbb{E} (log(p(x|y))) $$

Cross Entropy: Same as the maximum likelihood

Advantage:
- Well defined loss function as long as distribution $P$ is defined

Disadvantage:
- Can derive unlimited reward in some cases (Behaves like logistic)
