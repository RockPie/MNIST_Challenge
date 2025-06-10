import numpy as np

def ReLu(x):
    return np.maximum(0, x)

def der_ReLu(x):
    return np.where(x > 0, 1, 0)

def leaky_ReLu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def der_leaky_ReLu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)