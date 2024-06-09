import numpy as np

# Sigmoid (Logistic) Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh (Hyperbolic Tangent) Activation Function
def tanh(x):
    return np.tanh(x)

# ReLU (Rectified Linear Unit) Activation Function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

# ELU (Exponential Linear Unit) Activation Function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Softmax Activation Function
def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return exps / np.sum(exps, axis=0)

# Swish Activation Function
def swish(x):
    return x * sigmoid(x)

# Example usage:
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("ELU:", elu(x))
print("Softmax:", softmax(x))
print("Swish:", swish(x))
