import numpy as np
import struct, sys, tqdm, time
import matplotlib.pyplot as plt
from loguru import logger
from sj_lib_ML import *

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
            # logger.info(f"Magic number: {magic}, Size: {size}, Rows: {nrows}, Columns: {ncols}")
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        elif magic == 2049:
            # logger.info(f"Magic number: {magic}, Size: {size}")
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
def init_NN_parameters_random():
    layer_1_size = 32
    layer_2_size = 32
    layer_3_size = 16
    layer_4_size = 10

    w1 = np.random.randn(layer_1_size, 28 * 28) - 0.5
    b1 = np.random.randn(layer_1_size, 1) - 0.5
    w2 = np.random.randn(layer_2_size, layer_1_size) - 0.5
    b2 = np.random.randn(layer_2_size, 1) - 0.5
    w3 = np.random.randn(layer_3_size, layer_2_size) - 0.5
    b3 = np.random.randn(layer_3_size, 1) - 0.5
    w4 = np.random.randn(layer_4_size, layer_3_size) - 0.5
    b4 = np.random.randn(layer_4_size, 1) - 0.5

    return w1, b1, w2, b2, w3, b3, w4, b4

def init_NN_parameters_weighted():
    layer_1_size = 32
    layer_2_size = 16
    layer_3_size = 16
    layer_4_size = 10

    sigma = 1

    w1 = np.random.randn(layer_1_size, 28 * 28) - 0.5
    b1 = np.random.randn(layer_1_size, 1) - 0.5
    w2 = np.random.randn(layer_2_size, layer_1_size) - 0.5
    b2 = np.random.randn(layer_2_size, 1) - 0.5
    w3 = np.random.randn(layer_3_size, layer_2_size) - 0.5
    b3 = np.random.randn(layer_3_size, 1) - 0.5
    w4 = np.random.randn(layer_4_size, layer_3_size) - 0.5
    b4 = np.random.randn(layer_4_size, 1) - 0.5

    weight_layer_1 = np.sqrt(sigma / (28 * 28 + layer_2_size))
    weight_layer_2 = np.sqrt(sigma / (layer_1_size + layer_3_size))
    weight_layer_3 = np.sqrt(sigma / (layer_2_size + layer_4_size))

    w1 *= weight_layer_1
    b1 *= weight_layer_1
    w2 *= weight_layer_2
    b2 *= weight_layer_2
    w3 *= weight_layer_3
    b3 *= weight_layer_3

    return w1, b1, w2, b2, w3, b3, w4, b4

def update_NN_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    w4 -= learning_rate * dw4
    b4 -= learning_rate * db4
    return w1, b1, w2, b2, w3, b3

def update_NN_parameters_limited(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, learning_rate):
    # Limit the updates to a maximum of 0.1
    max_update = 0.5
    w1 -= np.clip(learning_rate * dw1, -max_update, max_update)
    b1 -= np.clip(learning_rate * db1, -max_update, max_update)
    w2 -= np.clip(learning_rate * dw2, -max_update, max_update)
    b2 -= np.clip(learning_rate * db2, -max_update, max_update)
    w3 -= np.clip(learning_rate * dw3, -max_update, max_update)
    b3 -= np.clip(learning_rate * db3, -max_update, max_update)
    w4 -= np.clip(learning_rate * dw4, -max_update, max_update)
    b4 -= np.clip(learning_rate * db4, -max_update, max_update)
    return w1, b1, w2, b2, w3, b3, w4, b4

def forward_propagation(X, w1, b1, w2, b2, w3, b3, w4, b4, use_batch_norm=False):
    z1 = w1.dot(X) + b1
    a1 = leaky_ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = leaky_ReLu(z2)
    z3 = w3.dot(a2) + b3
    a3 = leaky_ReLu(z3)
    z4 = w4.dot(a3) + b4
    a4 = softmax(z4)  # Output layer with softmax activation
    return z1, a1, z2, a2, z3, a3, z4, a4

def back_propagation_adam(X, Y, z1, a1, z2, a2, z3, a3, w1, w2, w3, dw1_last=None, db1_last=None, dw2_last=None, db2_last=None, dw3_last=None, db3_last=None, beta=0.90):
    m = Y.size
    # dZ3 = 2*(a3 - Y) * der_leaky_ReLu(z3)
    dZ3 = a3 - Y 
    dw3 = dZ3.dot(a2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2 = w3.T.dot(dZ3) * der_leaky_ReLu(z2)
    dw2 = dZ2.dot(a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = w2.T.dot(dZ2) * der_leaky_ReLu(z1)
    dw1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Adam update
    dw1 = beta * dw1_last + (1 - beta) * dw1 if dw1_last is not None else dw1
    db1 = beta * db1_last + (1 - beta) * db1 if db1_last is not None else db1
    dw2 = beta * dw2_last + (1 - beta) * dw2 if dw2_last is not None else dw2
    db2 = beta * db2_last + (1 - beta) * db2 if db2_last is not None else db2
    dw3 = beta * dw3_last + (1 - beta) * dw3 if dw3_last is not None else dw3
    db3 = beta * db3_last + (1 - beta) * db3 if db3_last is not None else db3

    return dw1, db1, dw2, db2, dw3, db3

def back_propagation_momentum(X, Y, z1, a1, z2, a2, z3, a3, z4, a4, w1, w2, w3, w4, dw1_last=None, db1_last=None, dw2_last=None, db2_last=None, dw3_last=None, db3_last=None, dw4_last=None, db4_last=None, beta=0.90):
    m = Y.shape[1]
    dZ4 = a4 - Y  # Derivative of the loss with respect to the output layer
    dw4 = dZ4.dot(a3.T) / m
    db4 = np.sum(dZ4, axis=1, keepdims=True) / m

    dZ3 = w4.T.dot(dZ4) * der_leaky_ReLu(z3)
    dw3 = dZ3.dot(a2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2 = w3.T.dot(dZ3) * der_leaky_ReLu(z2)
    dw2 = dZ2.dot(a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = w2.T.dot(dZ2) * der_leaky_ReLu(z1)
    dw1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Momentum update
    dw1 = beta * dw1_last + (1 - beta) * dw1 if dw1_last is not None else dw1
    db1 = beta * db1_last + (1 - beta) * db1 if db1_last is not None else db1
    dw2 = beta * dw2_last + (1 - beta) * dw2 if dw2_last is not None else dw2
    db2 = beta * db2_last + (1 - beta) * db2 if db2_last is not None else db2
    dw3 = beta * dw3_last + (1 - beta) * dw3 if dw3_last is not None else dw3
    db3 = beta * db3_last + (1 - beta) * db3 if db3_last is not None else db3
    dw4 = beta * dw4_last + (1 - beta) * dw4 if dw4_last is not None else dw4
    db4 = beta * db4_last + (1 - beta) * db4 if db4_last is not None else db4

    return dw1, db1, dw2, db2, dw3, db3, dw4, db4

def loss_function(Y, Y_hat):
    # return loss_func_categorical_crossentropy(Y, Y_hat)
    return loss_func_squared(Y, Y_hat)  # Using squared loss for simplicity

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()
    
    logger.debug(f"Training data shape: {train_data.shape}")
    logger.debug(f"Training labels shape: {train_labels.shape}")
    logger.debug(f"Test data shape: {test_data.shape}")
    logger.debug(f"Test labels shape: {test_labels.shape}")

    number_of_plots = 10
    target_epoch            = 1500
    target_train_accuracy   = 0.0
    target_test_accuracy    = 0.0
    target_train_loss       = 0.0
    target_test_loss        = 0.0

    epochs          = 2500
    learning_rate_max = 0.5
    learning_rate_min = 0.01
    batch_size      = 200
    momentum_beta   = 0.90

    # create batches of data
    train_data_batches = np.array_split(train_data, len(train_data) // batch_size)
    train_labels_batches = np.array_split(train_labels, len(train_labels) // batch_size)
    test_data_batches = np.array_split(test_data, len(test_data) // batch_size)
    test_labels_batches = np.array_split(test_labels, len(test_labels) // batch_size)

    # Flatten and normalize data
    train_data = train_data.reshape(train_data.shape[0], -1).T / 255.0    # (784,60000)
    test_data  = test_data.reshape(test_data.shape[0],  -1).T / 255.0

    # One-hot encode labels
    def one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels.flatten()].T

    train_labels_oh = one_hot(train_labels)  # (10,60000)
    test_labels_oh  = one_hot(test_labels)   # (10,10000)

    # Split data into batches
    train_data_batches  = np.array_split(train_data,  train_data.shape[1]//batch_size, axis=1)
    train_labels_batches= np.array_split(train_labels_oh, train_labels_oh.shape[1]//batch_size, axis=1)
    test_data_batches   = np.array_split(test_data,   test_data.shape[1]//batch_size,  axis=1)
    num_test_batches = test_labels_oh.shape[1] // batch_size
    test_labels_batches = np.array_split(test_labels_oh, num_test_batches, axis=1)

    epochs_monitor_loss = np.zeros(epochs)
    epochs_monitor_test_loss = np.zeros(epochs)
    epochs_monitor_accuracy = np.zeros(epochs)
    epochs_monitor_test_accuracy = np.zeros(epochs)

    w1, b1, w2, b2, w3, b3, w4, b4 = init_NN_parameters_weighted()
    dw1_last = np.zeros_like(w1)
    db1_last = np.zeros_like(b1)
    dw2_last = np.zeros_like(w2)
    db2_last = np.zeros_like(b2)
    dw3_last = np.zeros_like(w3)
    db3_last = np.zeros_like(b3)
    dw4_last = np.zeros_like(w4)
    db4_last = np.zeros_like(b4)

    # make the progress bar float
    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs", unit="epoch", ncols=100, leave=True, dynamic_ncols=True, position=0, file=sys.stdout):

        learning_rate = linear_learning_rate(epoch, epochs, initial_lr=learning_rate_max, final_lr= learning_rate_min)

        z1_epoch, a1_epoch, z2_epoch, a2_epoch, z3_epoch, a3_epoch, z4_epoch, a4_epoch = forward_propagation(train_data, w1, b1, w2, b2, w3, b3, w4, b4)

        z1_test_epoch, a1_test_epoch, z2_test_epoch, a2_test_epoch, z3_test_epoch, a3_test_epoch, z4_test_epoch, a4_test_epoch = forward_propagation(test_data, w1, b1, w2, b2, w3, b3, w4, b4)

        loss        = loss_function(train_labels_oh, a4_epoch)
        loss_test   = loss_function(test_labels_oh, a4_test_epoch)
        accuracy_train  = accuracy(train_labels_oh, a4_epoch)
        accuracy_test   = accuracy(test_labels_oh, a4_test_epoch)

        epochs_monitor_loss[epoch] = loss
        epochs_monitor_test_loss[epoch] = loss_test
        epochs_monitor_accuracy[epoch] = accuracy_train
        epochs_monitor_test_accuracy[epoch] = accuracy_test

        if epoch == target_epoch:
            target_train_accuracy = accuracy_train
            target_test_accuracy = accuracy_test
            target_train_loss = loss
            target_test_loss = loss_test

        # print the loss and accuracy every 100 epochs
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: -- Training Loss: {loss:.4f}, Accuracy: {accuracy_train*100:.2f}% -- Test Loss: {loss_test:.4f}, Accuracy: {accuracy_test*100:.2f}%")

        for batch_index in range(len(train_data_batches)):
            z1, a1, z2, a2, z3, a3, z4, a4 = forward_propagation(train_data_batches[batch_index], w1, b1, w2, b2, w3, b3, w4, b4)

            loss = loss_function(train_labels_batches[batch_index], a4)

            dw1, db1, dw2, db2, dw3, db3, dw4, db4 = back_propagation_momentum(train_data_batches[batch_index], train_labels_batches[batch_index], z1, a1, z2, a2, z3, a3, z4, a4, w1, w2, w3, w4, dw1_last=dw1_last, db1_last=db1_last, dw2_last=dw2_last, db2_last=db2_last, dw3_last=dw3_last, db3_last=db3_last, dw4_last=dw4_last, db4_last=db4_last, beta=momentum_beta)

            dw1_last, db1_last = dw1, db1
            dw2_last, db2_last = dw2, db2
            dw3_last, db3_last = dw3, db3
            dw4_last, db4_last = dw4, db4
            w1, b1, w2, b2, w3, b3, w4, b4 = update_NN_parameters_limited(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, learning_rate)

    logger.info(f"Final Training Loss: {loss:.4f}, Final Test Loss: {loss_test:.4f}")
    logger.info(f"Final Training Accuracy: {accuracy_train*100:.2f}%, Final Test Accuracy: {accuracy_test*100:.2f}%")

    # Plotting the loss over epochs
    plt.figure(figsize=(12, 6))
    # two axis
    plt.subplot(1, 2, 1)
    plt.plot(epochs_monitor_loss, label='Training Loss', color='blue')
    plt.plot(epochs_monitor_test_loss, label='Test Loss', color='orange')
    plt.axvline(x=target_epoch, color='red', linestyle='--', label=f'Target Epoch {target_epoch}')
    plt.axhline(y=target_train_loss, color='blue', linestyle='--', label=f'Target Train Loss {target_train_loss:.4f}')
    plt.axhline(y=target_test_loss, color='orange', linestyle='--', label=f'Target Test Loss {target_test_loss:.4f}')
    # log scale
    # plt.yscale('log')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.tight_layout()

    # Plotting the accuracy over epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs_monitor_accuracy, label='Training Accuracy', color='green')
    plt.plot(epochs_monitor_test_accuracy, label='Test Accuracy', color='red')
    plt.axvline(x=target_epoch, color='red', linestyle='--', label=f'Target Epoch {target_epoch}')
    plt.axhline(y=target_train_accuracy, color='green', linestyle='--', label=f'Target Train Accuracy {target_train_accuracy*100:.2f}%')
    plt.axhline(y=target_test_accuracy, color='red', linestyle='--', label=f'Target Test Accuracy {target_test_accuracy*100:.2f}%')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()

    plt_name = 'dump/res_' + time.strftime("%Y%m%d-%H%M%S") + ".png"
    plt.savefig(plt_name)