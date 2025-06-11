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

def der_softmax(x):
    s = softmax(x)
    return s * (1 - s)

def accuracy(Y, Y_hat):
    correct_predictions = np.sum(np.argmax(Y, axis=0) == np.argmax(Y_hat, axis=0))
    return correct_predictions / float(Y.shape[1])

def loss_func_log(Y, Y_hat):
    epsilon = 1e-15
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=0))

def loss_func_squared(Y, Y_hat):
    return 2* np.mean(np.sum((Y - Y_hat) ** 2, axis=0))/ Y.shape[1]

def loss_func_categorical_crossentropy(Y, Y_hat):
    epsilon = 1e-15
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=0))

def exp_learning_rate(epoch_current, epoch_total, initial_lr=5, final_lr=0.1, decay_rate=0.1):
    if epoch_current >= epoch_total:
        return final_lr
    else:
        return initial_lr * np.exp(-decay_rate * (epoch_current / epoch_total))