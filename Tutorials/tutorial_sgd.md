# Stochastic Gradient Descent and Backpropagation

## By Kiyeob Lee and Devarsh Jhaveri

## Introduction to Deep Learning
Deep Learning has driven remarkable improvements in recent years and still is one of the rapidly evolving areas in image classification, object detection, Natural Language Processing, and etc. Deep Learning requires a horrendously large(huge) training dataset, and thus in order to train all the dataset in neural network, there needs an algorithm that the network(net) is capable of learning all the dataset while training. Backpropgation perfectly fits in this purpose and Stochastic Gradient Descent(SGD) is one angle of the picture. This tutorial is to understand about how Backpropgation works using SGD. Throughout the tutorial, for simplicity, we will consider fully connected neural network, and all activation functions are sigmoid functions, which can be dropped, but it simplifies the tutorial. There are many other activations such as Rectified Linear Units(ReLU), Softmax, Maxpooling, and etc.

Our goal is to start with a simple neural network, and describe feed-forward, and finish with backpropagation algorithm.
### Contents
1. A simple neural network
2. Feed-forward
3. Backpropagation


### 1. A simple neural network(Fully connected neural network)
A neural network is a graph where there are three components of layers: Input, Hidden, and Output layers. Within each layer, there are nodes called neurons which are also called activation functions. Unlike fancy neural networks such as Recurrent Neural Network(RNN) or Long-Short-Term-Memory(LSTM), only adjacent layers are connected to each other, i.e., layers $l_{i-1}$ and $$l_{i+1}$$ are not directly connected, but connected only through $$l_{i}$$. For RNN and LSTM, the architecture is slightly different that $$l_{i-1}$$ and $$l_{i+1}$$ could be directly connected, but slight variations of Backpropagation works well in practice, and the philosophy is the same. Activation function is simply a function that takes an input and computes it's output which can be both linear and nonlinear. This offers great flexibility that a simple neural network that all activation functions are linear performs exactly the same as Principal Component Analysis, and if activation functions are nonlinear, it performs dimensionality redunction in nonlinear fashion.

### 2. Feed-forward
Once a training data instance $$x = (x_{1}, \cdots, x_{n})$$ comes into the input layer, each neuron in input layer sends an element $$x_{i}$$ to the first hidden layer $$l_{1}$$, and thus, hidden layer $$l_{1}$$ receives an array x. Provided that there is some initialization of weights $$w = (w_{1}, \cdots , w_{n})$$ in the hidden layers, neurons in layer $$l_{1}$$ takes a dot product $$\sum_{i=1}^{n}w_{i}x_{i}$$ which is a sclar and we call this a pre-activation ($$z^l$$) to the neuron since this is what goes into the activation. Note that if all weights in a hidden layer are all the same, then it would be a boring neural network, and thus, random initialization would make the neural network more interesting. Now, inductively, the first hidden layer $$l_{1}$$ 'feed-forwards' it's elements to the next layer until it reaches to the output layer.

Once the output layer receives values $$\hat{y}$$, it then it knows how far the true label $$y$$ and the predicted label $$\hat{y}$$, and based on a measure of distance between $$y$$ and $$\hat{y}$$, it updates how 'weights' in the neural network are supposd to be updated. This 'weight update' process is called backpropagation which will be described shortly.

Up to now, we considered only one training instance, but, it similarly repeates for all training instances and multiple iterations of the dataset until it minimizes a distance between $$y$$ and $$\hat{y}$$ for testing dataset.

### 3. Backpropagation
The backpropagation algorithm is divided into 2 parts as follows :
1. Propagation of Errors
2. Updating the weights.

#### Propagation of Errors

1. The activation of each neuron is the sigmoid function and the activation of each neuron in layer $$l$$ is given by $$a^l = \sigma(z^l)$$ for $$l=2,3,...,L$$. 
2. The output error is calculated by $$\delta^L = \nabla_a C \odot \sigma'(z^L)$$. 
3. Now this error is propagated backwards so that each and every neuron in the network has their respective error component to the output error. This in turn in as integral part of the algorithm. The error is backpropagated by the following equation. For each $$l=L-1,L-2,...,L$$ we have $$\delta^l = ((w^{l+1})^T\delta^{l+1}) \odot \sigma'(z^L)$$

#### Updating the weights
1. The gradient of the costfunction w.r.t the weights is calculated, this givus us an idea of how the cost function changes if the weights are changed. The gradient is claculated as follows: For each layer $$l=2,3,...,L$$ and $$i=1,2,...,n \in l$$ we have $$\frac{\partial C}{\partial w_i^l} = a_i^{l-1}\delta^l$$.
2. Now, the weights are updated using the above values in the following manner: For each $$l=L,L-1,L-2,...,2$$ we have $$w^l \rightarrow w^l-\alpha \frac{\partial C}{\partial w^l}$$

This complete one iteration of the backpropagation algorithm and the weights have been updated throughout the network.

Now the 2nd training example goes through the network and the same process is repeated until all the training examples have been sent through the network.

### Summary

Backpropagation is one of the most important algorithms in machine learning and it is used almost universally to solve ML problems. 
At the lower level it is an algorithm which helps us calculate derivatives quickly, which in turn are used by ANN as a learning algorithm in order to calculate the gradient descent w.r.t to weights.



### Other resources
There are many resources to learn about Backprop, but we found the belows the most useful.

https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/

https://www.youtube.com/watch?v=Ilg3gGewQ5U

https://www.youtube.com/watch?v=tIeHLnjs5U8



