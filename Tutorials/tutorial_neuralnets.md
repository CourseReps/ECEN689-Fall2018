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
Ability to implicitly detect complex nonlinear relationships between dependent and independent variables
Ability to detect all possible interactions between predictor variables
Availability of multiple training algorithms
Disadvantages:
Black box nature
Greater computational burden
Proneness to overfitting

