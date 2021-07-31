# NeuralNetwork

In this program, a 2-layer perceptron neural network is used to categorise hand-written digits. Numpy library is used and backpropagation has been implemented too. For the hidden layer, sigmoid was chosen as the activation function while softmax function was used in the output layer. Any dataset online can be used for hand-written digits. 

To increase optimization, various values were experimented for attributes like number of hidden layers, weights and bias. During backpropagation, gradients for 1st activation layer and 2nd input layer are constantly updated. Initial bias and weight parameters are then updated using computed gradient. 

The result for second activation layer is updated using the updated weight and bias parameters. The result for the second activation layer will be the predicted result.

A csv file can then be generated to store the predicted results. The attributes can be reconfigured to speed up training time.
