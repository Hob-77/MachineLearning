import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import reluactivation
import denselayer

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 ouput values
dense1 = denselayer.Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = reluactivation.Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# let's see output of the first few samples:
print(activation1.output[:5])
