# Introduction to Tensorflow

### By Ashish Kumar and Shabarish Prasad


## What is Tensorflow
Tensorflow is an open-source software library for dataflow programming, developed by Google Brain. Orginally Tensorflow was developed for performing heavy numeric computations. Now it is being widely used for the implementation of deep neural networks, since they involve heavy numerial computations. 

## Need for Tensorflow
Today, Tensorflow is the most preferred platform for the implementation of neural networks. Tensorflow provides both C++ and Python APIs. Tensorflow has a C/C++ backend and hence it is faster than pythonscript, even while using Tensorflow API on Python. Tensorflow architecture also supports distributed computing allowing the program to run across CPUs, GPUs and TPUs even across multiple servers to attain faster computing speeds.

## Working with Tensorflow
Programming in Tensorflow consists of two basic steps:
- Building a graph
- Executing the graph

![Tensorflow structure overview](https://drive.google.com/uc?id=1qw0-59tfCMavofamJ9wmQtoH5SFg_ZIl) 
The basic structure of a tensorflow program is illustrated in the diagram.

### Tensors
A tensor is an array of elements attributed by a rank, that defines the dimensonality of the tensor. For example, a tensor of rank 1 will be a scalar, a tensor of rank 2 will be a vector or a 1-dimensional array, a tensor of rank 2 will be a matrix and so on.

![Tensors of different dimensionality](https://drive.google.com/uc?id=1mKA9zbBaM2PCJ7jqj-JVYarRsspK-ajn)

### Session
A session brings a tensorflow code to life. All tensors are abstract outside the session. And the dataflow graphs can be run only inside the session. An illustration of the effect of the tensorflow session is given below:

```sh
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)

print([a,b])
```
>[<tf.Tensor 'Const:0' shape=() dtype=float32>, <tf.Tensor 'Const:0' shape=() dtype=float32>]

```sh
sess = tf.Session()
print(sess.run([a,b])
```
>[5.0, 6.0]

In the example, we can see that the tensor are able to retrive their value only when they are called from inside the session.

### Constants, Placeholders and Variable
Tensors are of three types in tensorflow.

Constants are the tensors that get initialized with a value and these values do not change until the end of the session.

```sh
import tensorflow as tf

a = tf.constant(5.0, dtype=tf.float32)
b = tf.constant(6.0, dtype=tf.float32)
```

Placeholders are promised a value at the time of execution. Like constants, once the placeholders are assigned a value, it does not change until the session ends.

```sh
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sess = tf.Session()
print(sess.run([a,b],{a:5.0, b:6.0}))
```
>[5.0, 6.0]

Variables are the tensors, the values in which keep changing through the course of the session. Unlike placeholders, variables in tensorflow should always be initialized with a value. And whenever variables are used in a tensorflow program, they should be initialized with the global_variables_initializer() function.An illustration can be seen below:
```sh
import tensorflow as tf

a = tf.Variable([1], tf.float32)
inc = tf.assign_add(a,[1])

sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(inc)
    
print(sess.run(a))
```
>[11]

### Computational Graph
In Tensorflow all programs are represented as dataflow graphs. In the graphs, each node represents an operation and all the edges represent the tensors. An example graph is shown below:

![Example Computational graph](https://drive.google.com/uc?id=1eUUYziXUR76t0kOpWvE4SC97Ce18PN0o)

These graphs are the way in which tensorflow builds the programming logic in the memory.

## Implemenatation Example - Linear Regression
```sh
import tensorflow as tf

w = tf.Variables([0.3],tf.float32)
b = tf.Variables([0.3],tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_regression = w * x + b

sqaure_fn = tf.square(linear_regression - y)
loss = tf.reduce_sum(sqaure_fn)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3,-4]})
```
## Tensorboard
Tensorboard is a suite of visualization tools built inside tensorflow. This enables users to understand, debug and optimize tensorflow programs. An example output graph of tensorboard is shown below:

![Tensorboard graph](https://drive.google.com/uc?id=1uwq_v1E-XCI_P7_VsHBMnG6gj7KWRtf-)

## References
- https://www.tensorflow.org/tutorials/
- https://www.youtube.com/watch?v=yX8KuPZCAMo
- https://www.youtube.com/watch?v=MotG3XI2qSs
- https://www.tensorflow.org/deploy/distributed
- https://developers.google.com/machine-learning/crash-course/ml-intro





