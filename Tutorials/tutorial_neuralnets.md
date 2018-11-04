# Neural Netowrks
by Amrita Veerabagu and Divyank Garg

Neural Network is one the most important deep learning algorithm and its working can be considered as same as how our brain system work. As brain receives information from environment as input and then brain stimulate that with help of neurons and then give output to that information. Like that only the neural network is composed of a large number of highly interconnected processing elements (neurons) working in unison to solve specific problems. The neural network consist of 3 type of layers called- Input layer, Hidden layer and Output layer. The hidden layer can be of any number from 1 to many that depends on us.

•	X= Input points
•	W= Connector weight
•	b= Bias
•	f= activation function (many different types like- Sigmoid, Tanh, ReLU function)

Implementation of Neural Networks

1) Forward Pass: Refers to calculation process, values of the output layers from the inputs data. Its traversing through all neurons from first to last layer. In this the input is taken in input layer and then based on weightage of connector and bias value of neuron in hidden layer and activation function the value is output is found.
2) Backward Pass: Refers to process of counting changes in weights (de facto learning), using gradiend descent algorithm or similar. Computation is made from last layer, backward to the first layer. Based on minimization of gradient descent the new weightage is calculated for connector and this both forward and backward is continued until the cost function reches minimum value.

Advantages:

- Ability to implicitly detect complex nonlinear relationships between dependent and independent variables
- Ability to detect all possible interactions between predictor variables
- Availability of multiple training algorithms

Disadvantages:
- Black box nature
- Greater computational burden
- Proneness to overfitting

Variations of Neural Network:

- Fully Connected Neural Network
- Convolution Neural Network
- Recurrent Neural Network

Convolutional Neural Network(CNN):

CNNs are special type of neural networks used mainly when we have spatial data in our hand. It is mostly used in Computer vision tasks like Object detection, Object recognition. Image classification etc. 

When a computer sees an image (takes an image as input), it will see an array of pixel values. Depending on the resolution and size of the image, it will see a 32 x 32 x 3 array of numbers (The 3 refers to RGB values). The idea is that you give the computer this array of numbers and it will output numbers that describe the probability of the image being a certain class (.80 for cat, .15 for dog, .05 for bird, etc). A convolutional neural network looks for a specific feature or a characteristic from this array in each layer of the network. This way we could drastically reduce the number of weights that we have to find. The CNN is able perform image classification by looking for low level features such as edges and curves, and then building up to more abstract concepts through a series of convolutional layers. 

Therefore, the convolutional layers acts as a feature extraction module and the extracted features from the image will be sent to a fuly-connected neural network as predictors to perform the appropriate task.

Recurrent Neural Network(RNN):

The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps.

RNNs have shown great success in many NLP tasks. The most commonly used type of RNNs are LSTMs, which are much better at capturing long-term dependencies than vanilla RNNs are. A single neuron of a LSTM model consists of gates which help them to process the information they have in them in a slightly different way than the RNN. These gates input, output and forget gate are responsible for learning which information to retain and which information to forget in the course of time. This makes the LSTMs better at capturing long-term dependencies, making them better than the vanilla RNNs. 


