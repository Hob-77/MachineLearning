import numpy as np
import activationsoftmax

softmax = activationsoftmax.Activation_Softmax()

softmax.forward([[1, 2, 3]])
print(softmax.output)
softmax.forward([[0.5, 1, 1.5]])
print(softmax.output)
