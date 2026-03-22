import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import denselayer
import reluactivation
import activationsoftmax

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = denselayer.Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = reluactivation.Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = denselayer.Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = activationsoftmax.Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activaton function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])
