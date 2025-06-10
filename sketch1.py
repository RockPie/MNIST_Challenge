import numpy as np
import struct, sys, tqdm
import matplotlib.pyplot as plt
from loguru import logger

file_training_data      = 'data/train-images.idx3-ubyte'
file_training_labels    = 'data/train-labels.idx1-ubyte'
file_test_data          = 'data/t10k-images.idx3-ubyte'
file_test_labels        = 'data/t10k-labels.idx1-ubyte'

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} |{level:.1s}|: {message}", colorize=True, level="INFO")
logger.add(sys.stderr, format="{time:HH:mm:ss} |{level:.1s}|: {message}", colorize=True, level="DEBUG")
logger.add(sys.stderr, format="{time:HH:mm:ss} |{level:.1s}|: {message}", colorize=True, level="WARNING")

# * === Reading data files ==============================================================
# * =====================================================================================
def load_idx_data_file(file_path):
   with open(file_path, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        if magic == 2051:
            nrows, ncols = struct.unpack('>II', f.read(8))
            logger.info(f"Magic number: {magic}, Size: {size}, Rows: {nrows}, Columns: {ncols}")
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        elif magic == 2049:
            logger.info(f"Magic number: {magic}, Size: {size}")
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size,1))
        return data

def load_data():
    training_data = load_idx_data_file(file_training_data)
    training_labels = load_idx_data_file(file_training_labels)
    test_data = load_idx_data_file(file_test_data)
    test_labels = load_idx_data_file(file_test_labels)
    return (training_data, training_labels), (test_data, test_labels)

# * === NN Components ===================================================================
# * =====================================================================================
def init_NN_parameters():
    w1 = np.random.randn(16, 28 * 28) - 0.5
    b1 = np.random.randn(16, 1) - 0.5
    w2 = np.random.randn(16, 16) - 0.5
    b2 = np.random.randn(16, 1) - 0.5
    w3 = np.random.randn(10, 16) - 0.5
    b3 = np.random.randn(10, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

def update_NN_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3

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

def forward_propagation(X, w1, b1, w2, b2, w3, b3):
    z1 = w1.dot(X) + b1
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLu(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def back_propagation(X, Y, z1, a1, z2, a2, z3, a3, w1, w2, w3):
    m = Y.size
    dZ3 = 2*(a3 - Y) * der_ReLu(z3)
    dw3 = dZ3.dot(a2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2 = w3.T.dot(dZ3) * der_ReLu(z2)
    dw2 = dZ2.dot(a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = w2.T.dot(dZ2) * der_ReLu(z1)
    dw1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dw1, db1, dw2, db2, dw3, db3

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()
    
    logger.debug(f"Training data shape: {train_data.shape}")
    logger.debug(f"Training labels shape: {train_labels.shape}")
    logger.debug(f"Test data shape: {test_data.shape}")
    logger.debug(f"Test labels shape: {test_labels.shape}")

    number_of_plots = 10
    
    # Show the first 10 training images
    # plt.figure(figsize=(10, 4))
    # for i in range(number_of_plots):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(train_data[i].reshape(28, 28), cmap='gray')
    #     plt.title(f"Label: {train_labels[i][0]}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # Flatten and normalize data
    train_data = train_data.reshape(train_data.shape[0], -1).T / 255.0
    test_data = test_data.reshape(test_data.shape[0], -1).T / 255.0

    # One-hot encode labels
    def one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels.flatten()].T

    train_labels_oh = one_hot(train_labels)
    test_labels_oh = one_hot(test_labels)

    epochs = 1500
    epochs_monitor_loss = np.zeros(epochs)
    epochs_monitor_test_loss = np.zeros(epochs)
    learning_rate = 0.01
    w1, b1, w2, b2, w3, b3 = init_NN_parameters()

    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        z1, a1, z2, a2, z3, a3 = forward_propagation(train_data, w1, b1, w2, b2, w3, b3)
        z1_test, a1_test, z2_test, a2_test, z3_test, a3_test = forward_propagation(test_data, w1, b1, w2, b2, w3, b3)
        # Cross-entropy loss
        loss = -np.mean(np.sum(train_labels_oh * np.log(a3 + 1e-8), axis=0))
        loss_test = -np.mean(np.sum(test_labels_oh * np.log(a3_test + 1e-8), axis=0))
        epochs_monitor_loss[epoch] = loss
        epochs_monitor_test_loss[epoch] = loss_test
        dw1, db1, dw2, db2, dw3, db3 = back_propagation(train_data, train_labels_oh, z1, a1, z2, a2, z3, a3, w1, w2, w3)
        w1, b1, w2, b2, w3, b3 = update_NN_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_monitor_loss, label='Training Loss', color='blue')
    plt.plot(epochs_monitor_test_loss, label='Test Loss', color='orange')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()