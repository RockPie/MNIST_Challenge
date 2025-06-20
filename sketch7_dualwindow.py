import numpy as np                  # the only computation lib!
import struct, time                 # struct for reading idx files, time for naming the output file
import matplotlib.pyplot as plt     # plotting the results
from sj_lib_ML import *             # my own library with useful functions
import json                         # for saving the parameters

file_training_data      = 'data/train-images.idx3-ubyte'
file_training_labels    = 'data/train-labels.idx1-ubyte'
file_test_data          = 'data/t10k-images.idx3-ubyte'
file_test_labels        = 'data/t10k-labels.idx1-ubyte'

# * === Reading data files ==============================================================
# * =====================================================================================
def load_idx_data_file(file_path):
   with open(file_path, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        if magic == 2051:
            nrows, ncols = struct.unpack('>II', f.read(8))
            # print(f"Magic number: {magic}, Size: {size}, Rows: {nrows}, Columns: {ncols}")
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        elif magic == 2049:
            # print(f"Magic number: {magic}, Size: {size}")
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
# * ==============================================================
img_size     = 28
local_block_size_l1  = 3 # unit of pixels
local_block_step_l1  = 1 # unit of pixels
local_block_size_l1b = 6 # unit of pixels
local_block_step_l1b = 2 # unit of pixels
local_block_size_l2  = 3 # unit of local blocks on layer 1
local_block_step_l2  = 2 # unit of local blocks on layer 1
layer_2b_size = 64
layer_3_size  = 128
layer_4_size  = 10

def init_NN_parameters_weighted_local():
    layer_1_size = ((img_size - local_block_size_l1) // local_block_step_l1 + 1) ** 2
    layer_1b_size = ((img_size - local_block_size_l1b) // local_block_step_l1b + 1) ** 2
    layer_2_size = ((int(np.sqrt(layer_1_size)) - local_block_size_l2) // local_block_step_l2 + 1) ** 2

    print(" --- Neural Network Parameters -------------------------")
    print(f"Layer 1 size: {layer_1_size}, Layer 1b size: {layer_1b_size}")
    print(f"Layer 2 size: {layer_2_size}, Layer 2b size: {layer_2b_size}")
    print(f"Layer 3 size: {layer_3_size}, Layer 4 size: {layer_4_size}")
    # layer 1b is for larger local blocks
    total_neurons = layer_1_size + layer_2_size + layer_3_size + layer_4_size + layer_1b_size + layer_2b_size
    print(f"Total neurons: {total_neurons}")
    total_parameters = (layer_1_size * img_size * img_size) + (layer_2_size * layer_1_size) + (layer_3_size * (layer_2_size + layer_2b_size)) + (layer_4_size * layer_3_size) + (layer_1b_size * img_size * img_size) + (layer_2b_size * layer_1b_size)
    print(f"Total parameters: {total_parameters}")
    print("--------------------------------------------------------")

    sigma = 3

    # Initialize weights and biases by fan_in and fan_out method
    w1 = np.zeros((layer_1_size, img_size * img_size))
    idx1 = 0
    for row in range(0, img_size - local_block_size_l1 + 1, local_block_step_l1):
        for col in range(0, img_size - local_block_size_l1 + 1, local_block_step_l1):
            for dy in range(local_block_size_l1):
                for dx in range(local_block_size_l1):
                    y = row + dy
                    x = col + dx
                    idx = y * img_size + x
                    w1[idx1, idx] = np.random.randn() - 0.5
            idx1 += 1
    b1 = np.random.randn(layer_1_size, 1) - 0.5

    w2 = np.zeros((layer_2_size, layer_1_size))
    side_1 = int(np.sqrt(layer_1_size))
    idx2 = 0
    for row in range(0, side_1 - local_block_size_l2 + 1, local_block_step_l2):
        for col in range(0, side_1 - local_block_size_l2 + 1, local_block_step_l2):
            for dy in range(local_block_size_l2):
                for dx in range(local_block_size_l2):
                    y = row + dy
                    x = col + dx
                    idx = y * side_1 + x
                    w2[idx2, idx] = np.random.randn() - 0.5
            idx2 += 1
    b2 = np.random.randn(layer_2_size, 1) - 0.5
    w3 = np.random.randn(layer_3_size, layer_2_size + layer_2b_size) - 0.5
    b3 = np.random.randn(layer_3_size, 1) - 0.5
    w4 = np.random.randn(layer_4_size, layer_3_size) - 0.5
    b4 = np.random.randn(layer_4_size, 1) - 0.5

    w1b = np.zeros((layer_1b_size, img_size * img_size))
    idx1b = 0
    for row in range(0, img_size - local_block_size_l1b + 1, local_block_step_l1b):
        for col in range(0, img_size - local_block_size_l1b + 1, local_block_step_l1b):
            for dy in range(local_block_size_l1b):
                for dx in range(local_block_size_l1b):
                    y = row + dy
                    x = col + dx
                    idx = y * img_size + x
                    w1b[idx1b, idx] = np.random.randn() - 0.5
            idx1b += 1
    b1b = np.random.randn(layer_1b_size, 1) - 0.5

    w2b = np.zeros((layer_2b_size, layer_1b_size))
    side_1b = int(np.sqrt(layer_1b_size))
    idx2b = 0
    for row in range(0, side_1b - local_block_size_l2 + 1, local_block_step_l2):
        for col in range(0, side_1b - local_block_size_l2 + 1, local_block_step_l2):
            for dy in range(local_block_size_l2):
                for dx in range(local_block_size_l2):
                    y = row + dy
                    x = col + dx
                    idx = y * side_1b + x
                    w2b[idx2b, idx] = np.random.randn() - 0.5
            idx2b += 1
    b2b = np.random.randn(layer_2b_size, 1) - 0.5

    weight_layer_1 = np.sqrt(sigma / (28 * 28 + layer_2_size))
    weight_layer_2 = np.sqrt(sigma / (layer_1_size + layer_3_size))
    weight_layer_3 = np.sqrt(sigma / (layer_2_size + layer_4_size + layer_2b_size))
    weight_layer_1b = np.sqrt(sigma / (28 * 28 + layer_2b_size))
    weight_layer_2b = np.sqrt(sigma / (layer_1b_size + layer_3_size))

    w1 *= weight_layer_1
    b1 *= weight_layer_1
    w2 *= weight_layer_2
    b2 *= weight_layer_2
    w3 *= weight_layer_3
    b3 *= weight_layer_3
    w1b *= weight_layer_1b
    b1b *= weight_layer_1b
    w2b *= weight_layer_2b
    b2b *= weight_layer_2b

    return w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b, total_neurons, total_parameters

def update_NN_parameters_limited(w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw1b, db1b, dw2b, db2b, learning_rate):
    max_update = 0.5 
    # 0.5 is not practically having any effect any more
    w1 -= np.clip(learning_rate * dw1, -max_update, max_update)
    b1 -= np.clip(learning_rate * db1, -max_update, max_update)
    w2 -= np.clip(learning_rate * dw2, -max_update, max_update)
    b2 -= np.clip(learning_rate * db2, -max_update, max_update)
    w3 -= np.clip(learning_rate * dw3, -max_update, max_update)
    b3 -= np.clip(learning_rate * db3, -max_update, max_update)
    w4 -= np.clip(learning_rate * dw4, -max_update, max_update)
    b4 -= np.clip(learning_rate * db4, -max_update, max_update)
    w1b -= np.clip(learning_rate * dw1b, -max_update, max_update)
    b1b -= np.clip(learning_rate * db1b, -max_update, max_update)
    w2b -= np.clip(learning_rate * dw2b, -max_update, max_update)
    b2b -= np.clip(learning_rate * db2b, -max_update, max_update)
    return w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b

def forward_propagation(X, w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b):
    leaky_rate = 1e-2 # universal leaky rate for all

    # --- Layer 1 (small window) ---
    z1 = w1.dot(X) + b1
    a1 = leaky_ReLu(z1, alpha=leaky_rate)
    # a1_drop, dropout_mask = dropout(a1, p=0.1)
    a1_drop = a1  # No dropout in this version

    # --- Layer 1b (large window) ---
    z1b = w1b.dot(X) + b1b
    a1b = leaky_ReLu(z1b, alpha=leaky_rate)
    # a1b_drop, dropout_mask_b = dropout(a1b, p=0.01)
    a1b_drop = a1b  # No dropout in this version

    # --- Layer 2 (small window) ---
    z2 = w2.dot(a1_drop) + b2
    a2 = leaky_ReLu(z2, alpha=leaky_rate)
    # a2_drop, dropout_mask = dropout(a2, p=0.01)
    a2_drop = a2  # No dropout in this version

    # --- Layer 2b (large window) ---
    z2b = w2b.dot(a1b_drop) + b2b
    a2b = leaky_ReLu(z2b, alpha=leaky_rate)
    # a2b_drop, dropout_mask_b = dropout(a2b, p=0.01)
    a2b_drop = a2b  # No dropout in this version

    # ! REMEMBER the order of the layers!!!!
    # print(f"Shape of a1: {a1.shape}, Shape of a1b: {a1b.shape}")
    # print(f"Shape of a2: {a2.shape}, Shape of a2b: {a2b.shape}")
    a2_combined = np.vstack((a2_drop, a2b_drop))
    # print(f"Shape of a2_combined: {a2_combined.shape}, Shape of w3: {w3.shape}")

    # --- Layer 3 (Aggregate) ---
    z3 = w3.dot(a2_combined) + b3
    a3 = leaky_ReLu(z3, alpha=leaky_rate)
    a3_drop, dropout_mask = dropout(a3, p=0.15)

    # --- Layer 4 (Output) ---
    z4 = w4.dot(a3_drop) + b4

    # 1.0 and 2.0 are tested to be worse
    softmax_temp = 3.0
    a4 = softmax(z4, softmax_temp)

    return z1, a1, z2, a2, z3, a3, z4, a4, z1b, a1b, z2b, a2b, dropout_mask, softmax_temp

def back_propagation_momentum(X, Y, z1, a1, z2, a2, z3, a3, z4, a4, z1b, a1b, z2b, a2b, w1, w2, w3, w4, w1b, w2b, dw1_last=None, db1_last=None, dw2_last=None, db2_last=None, dw3_last=None, db3_last=None, dw4_last=None, db4_last=None, dw1b_last=None, db1b_last=None, dw2b_last=None, db2b_last=None, beta=0.90, dropout_mask=None, softmax_temp=2.0):
    m = Y.shape[1]
    lambda_L2_reg = 0.05
    
    # --- Layer 4 (Output) ---
    dZ4 = a4 - Y
    if softmax_temp != 1.0:
        dZ4 *= softmax_temp
    dw4 = dZ4.dot(a3.T) / m
    # L2 regularization
    dw4 += lambda_L2_reg * w4 / m
    db4 = np.sum(dZ4, axis=1, keepdims=True) / m

    # --- Layer 3 (Aggregate) ---
    dZ3_raw = w4.T.dot(dZ4) * der_leaky_ReLu(z3)
    if dropout_mask is not None:
        dZ3_raw *= dropout_mask

    dZ3 = dZ3_raw

    a2_combined = np.vstack((a2, a2b))

    dw3 = dZ3.dot(a2_combined.T) / m
    dw3 += lambda_L2_reg * w3 / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    # --- Layer 2 (small window)  ---
    # --- Layer 2b (large window) ---
    dZ2_combined = w3.T.dot(dZ3)
    # Split dZ2_combined into dZ2 and dZ1b
    dZ2_raw = dZ2_combined[:w2.shape[0], :] * der_leaky_ReLu(z2)
    dZ2b_raw = dZ2_combined[w2.shape[0]:, :] * der_leaky_ReLu(z2b)
    
    dZ2 = dZ2_raw
    dw2 = dZ2.dot(a1.T) / m
    dw2 += lambda_L2_reg * w2 / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ2b = dZ2b_raw
    dw2b = dZ2b.dot(a1b.T) / m
    dw2b += lambda_L2_reg * w2b / m
    db2b = np.sum(dZ2b, axis=1, keepdims=True) / m

    # --- Layer 1 (small window) ---
    dZ1_raw = w2.T.dot(dZ2) * der_leaky_ReLu(z1)
    dZ1 = dZ1_raw

    dw1 = dZ1.dot(X.T) / m
    dw1 += lambda_L2_reg * w1 / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # --- Layer 1b (large window) ---
    dZ1b_raw = w2b.T.dot(dZ2b) * der_leaky_ReLu(z1b)
    dZ1b = dZ1b_raw

    dw1b = dZ1b.dot(X.T) / m
    dw1b += lambda_L2_reg * w1b / m
    db1b = np.sum(dZ1b, axis=1, keepdims=True) / m

    # --- Momentum Update ---
    dw1 = mom(dw1, dw1_last)
    db1 = mom(db1, db1_last)
    dw2 = mom(dw2, dw2_last)
    db2 = mom(db2, db2_last)
    dw3 = mom(dw3, dw3_last)
    db3 = mom(db3, db3_last)
    dw4 = mom(dw4, dw4_last)
    db4 = mom(db4, db4_last)
    dw1b = mom(dw1b, dw1b_last)
    db1b = mom(db1b, db1b_last)
    dw2b = mom(dw2b, dw2b_last)
    db2b = mom(db2b, db2b_last)

    return dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw1b, db1b, dw2b, db2b

# * === Loss Function ===================================================================
# * =====================================================================================
def loss_function(Y, Y_hat):
    # return loss_func_categorical_crossentropy(Y, Y_hat)
    return loss_func_squared(Y, Y_hat)
    # return loss_func_log(Y, Y_hat)

# * === Main Execution ==================================================================
# * =====================================================================================
if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()

    # rotate the training data
    number_of_rotations = 8
    rotation_limit = 24
    # create number_of_rotations copies of the training data, each rotated by a random angle
    train_data_rotated = []
    for i in range(number_of_rotations):
        angle = i * 2 * rotation_limit / (number_of_rotations - 1) - rotation_limit
        # ramdom x and y zoom factor between 0.8 and 1.2
        # zoom_factor_x = np.random.uniform(0.8, 1.2)
        # zoom_factor_y = np.random.uniform(0.8, 1.2)
        # print(f"Rotating training data by {angle:.2f} degrees...")
        rotated_data = np.array([rotate_image(image, angle) for image in train_data])
        # randomly offset the image by a random number of pixels in x and y direction
        offset_x = np.random.randint(-2, 2)
        offset_y = np.random.randint(-2, 2)
        rotated_data = np.array([image_offset(image, offset_y, offset_x) for image in rotated_data])
        train_data_rotated.append(rotated_data)

    number_of_plots         = 10
    epoch_print_every       = 10
    target_train_accuracy   = 0.0
    target_test_accuracy    = 0.0
    target_train_loss       = 0.0
    target_test_loss        = 0.0

    # it will run a little bit longer than target to check overfit
    epochs                  = 1600
    target_epoch            = 1500

    # learning rate related parameters
    learning_rate_max       = 0.04
    learning_rate_min       = 0.002
    learning_rate_saturate_epoch = int(0.2 * epochs)

    batch_size              = 128
    # > 128 can take all the CPU
    momentum_beta           = 0.9 
    # 0.8 and 0.95 is tested to perform worse

    # create batches of data
    train_data_batches = np.array_split(train_data, len(train_data) // batch_size)
    # train_smoothed_data_batches = np.array_split(train_smoothed_data, len(train_smoothed_data) // batch_size)
    train_data_rotated_batches = [np.array_split(rotated_data, len(rotated_data) // batch_size) for rotated_data in train_data_rotated]
    train_labels_batches = np.array_split(train_labels, len(train_labels) // batch_size)

    test_data_batches = np.array_split(test_data, len(test_data) // batch_size)
    test_labels_batches = np.array_split(test_labels, len(test_labels) // batch_size)

    # Flatten and normalize data
    train_data = train_data.reshape(train_data.shape[0], -1).T / 255.0
    train_rotated_data = [rotated_data.reshape(rotated_data.shape[0], -1).T / 255.0 for rotated_data in train_data_rotated]
    # train_smoothed_data = train_smoothed_data.reshape(train_smoothed_data.shape[0], -1).T / 255.0
    test_data  = test_data.reshape(test_data.shape[0],  -1).T / 255.0
    
    train_labels_oh = one_hot_eplison(train_labels)  # (10,60000)
    test_labels_oh  = one_hot_eplison(test_labels)   # (10,10000)

    # Split data into batches
    train_data_batches  = np.array_split(train_data,  train_data.shape[1]//batch_size, axis=1)
    train_labels_batches= np.array_split(train_labels_oh, train_labels_oh.shape[1]//batch_size, axis=1)
    train_rotated_data_batches = [np.array_split(rotated_data, rotated_data.shape[1]//batch_size, axis=1) for rotated_data in train_rotated_data]

    test_data_batches   = np.array_split(test_data,   test_data.shape[1]//batch_size,  axis=1)
    num_test_batches = test_labels_oh.shape[1] // batch_size
    test_labels_batches = np.array_split(test_labels_oh, num_test_batches, axis=1)

    epochs_monitor_loss = np.zeros(epochs)
    epochs_monitor_test_loss = np.zeros(epochs)
    epochs_monitor_accuracy = np.zeros(epochs)
    epochs_monitor_test_accuracy = np.zeros(epochs)

    target_train_accuracy_array = []
    target_test_accuracy_array = []
    target_train_loss_array = []
    target_test_loss_array = []

    w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b, total_neurons, total_parameters = init_NN_parameters_weighted_local()
    dw1_last = np.zeros_like(w1)
    db1_last = np.zeros_like(b1)
    dw2_last = np.zeros_like(w2)
    db2_last = np.zeros_like(b2)
    dw3_last = np.zeros_like(w3)
    db3_last = np.zeros_like(b3)
    dw4_last = np.zeros_like(w4)
    db4_last = np.zeros_like(b4)
    dw1b_last = np.zeros_like(w1b)
    db1b_last = np.zeros_like(b1b)
    dw2b_last = np.zeros_like(w2b)
    db2b_last = np.zeros_like(b2b)

    # make the progress bar float
    for epoch in range(epochs):

        # learning_rate = linear_learning_rate(epoch, epochs, initial_lr=learning_rate_max, final_lr= learning_rate_min)
        learning_rate = exp_learning_rate(epoch, learning_rate_saturate_epoch, initial_lr=learning_rate_max, final_lr=learning_rate_min)

        z1_epoch, a1_epoch, z2_epoch, a2_epoch, z3_epoch, a3_epoch, z4_epoch, a4_epoch, z1b_epoch, a1b_epoch, z2b_epoch, a2b_epoch, mask_dump, softmax_temp = forward_propagation(train_data, w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b)

        z1_test_epoch, a1_test_epoch, z2_test_epoch, a2_test_epoch, z3_test_epoch, a3_test_epoch, z4_test_epoch, a4_test_epoch, z1b_test_epoch, a1b_test_epoch, z2b_test_epoch, a2b_test_epoch, make_dump, softmax_temp_test = forward_propagation(test_data, w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b)

        loss        = loss_function(train_labels_oh, a4_epoch)
        loss_test   = loss_function(test_labels_oh, a4_test_epoch)
        accuracy_train  = accuracy(train_labels_oh, a4_epoch)
        accuracy_test   = accuracy(test_labels_oh, a4_test_epoch)

        epochs_monitor_loss[epoch] = loss
        epochs_monitor_test_loss[epoch] = loss_test
        epochs_monitor_accuracy[epoch] = accuracy_train
        epochs_monitor_test_accuracy[epoch] = accuracy_test

        if np.abs(epoch - target_epoch) < 50:
            target_train_accuracy_array.append(accuracy_train)
            target_test_accuracy_array.append(accuracy_test)
            target_train_loss_array.append(loss)
            target_test_loss_array.append(loss_test)

        # print the loss and accuracy
        if epoch % epoch_print_every == 0:
            # tqdm.write(f"[Epoch {epoch:03d}] Train Loss: {loss:.4f}, Acc: {accuracy_train*100:.2f}%, Test Loss: {loss_test:.4f}, Acc: {accuracy_test*100:.2f}%")
            print(f"[Epoch {epoch:03d}] Train Loss: {loss:.4f}, Acc: {accuracy_train*100:.2f}%, Test Loss: {loss_test:.4f}, Acc: {accuracy_test*100:.2f}%")

        # shuffle the batch order
        index_shuffle = np.random.permutation(len(train_data_batches))
        train_data_batches = [train_data_batches[i] for i in index_shuffle]
        # train_smoothed_data_batches = [train_smoothed_data_batches[i] for i in index_shuffle]
        train_labels_batches = [train_labels_batches[i] for i in index_shuffle]
        # Transpose, shuffle, then transpose back to maintain batch structure
        train_rotated_data_batches = list(zip(*train_rotated_data_batches))
        train_rotated_data_batches = [train_rotated_data_batches[i] for i in index_shuffle]
        train_rotated_data_batches = list(zip(*train_rotated_data_batches))

        for batch_index in range(len(train_data_batches)):
            random_choice_of_rotation = np.random.randint(0, number_of_rotations + 1)
            if random_choice_of_rotation % (number_of_rotations+1) == 0:
                # use the original data
                train_batch = train_data_batches[batch_index]
            else:
                # use the rotated data
                train_batch = train_rotated_data_batches[(random_choice_of_rotation % (number_of_rotations+1)) - 1][batch_index]

            train_labels_batch = train_labels_batches[batch_index]

            z1, a1, z2, a2, z3, a3, z4, a4, z1b, a1b, z2b, a2b, mask, softmax_temp = forward_propagation(train_batch, w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b)

            loss = loss_function(train_labels_batch, a4)

            dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw1b, db1b, dw2b, db2b = back_propagation_momentum(train_batch, train_labels_batch, z1, a1, z2, a2, z3, a3, z4, a4, z1b, a1b, z2b, a2b, w1, w2, w3, w4, w1b, w2b, dw1_last=dw1_last, db1_last=db1_last, dw2_last=dw2_last, db2_last=db2_last, dw3_last=dw3_last, db3_last=db3_last, dw4_last=dw4_last, db4_last=db4_last, dw1b_last=dw1b_last, db1b_last=db1b_last, dw2b_last=dw2b_last, db2b_last=db2b_last, beta=momentum_beta, dropout_mask=mask, softmax_temp=softmax_temp)

            dw1_last, db1_last = dw1, db1
            dw2_last, db2_last = dw2, db2
            dw3_last, db3_last = dw3, db3
            dw4_last, db4_last = dw4, db4
            dw1b_last, db1b_last = dw1b, db1b
            dw2b_last, db2b_last = dw2b, db2b

            w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b = update_NN_parameters_limited(w1, b1, w2, b2, w3, b3, w4, b4, w1b, b1b, w2b, b2b, dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw1b, db1b, dw2b, db2b, learning_rate)

    print(f"Final Training Loss: {loss:.4f}, Final Test Loss: {loss_test:.4f}")
    print(f"Final Training Accuracy: {accuracy_train*100:.2f}%, Final Test Accuracy: {accuracy_test*100:.2f}%")

    target_train_accuracy = np.mean(np.array(target_train_accuracy_array))
    target_test_accuracy = np.mean(np.array(target_test_accuracy_array))
    target_train_loss = np.mean(np.array(target_train_loss_array))
    target_test_loss = np.mean(np.array(target_test_loss_array))

    print(f"Target Train Accuracy: {target_train_accuracy:.4f}, Target Test Accuracy: {target_test_accuracy:.4f}")
    print(f"Target Train Loss: {target_train_loss:.4f}, Target Test Loss: {target_test_loss:.4f}")

    target_train_error_rate = 1 - target_train_accuracy
    target_test_error_rate = 1 - target_test_accuracy

    target_train_error_rate_err = np.std(np.array(target_train_accuracy_array))/ np.sqrt(len(target_train_accuracy_array))
    target_test_error_rate_err = np.std(np.array(target_test_accuracy_array)) / np.sqrt(len(target_test_accuracy_array))
    target_train_loss_err = np.std(np.array(target_train_loss_array)) / np.sqrt(len(target_train_loss_array))
    target_test_loss_err = np.std(np.array(target_test_loss_array)) / np.sqrt(len(target_test_loss_array))

    # Plotting the loss over epochs
    plt.figure(figsize=(12, 6))
    # two axis
    plt.subplot(1, 2, 1)
    plt.plot(epochs_monitor_loss, label='Training Loss', color='blue')
    plt.plot(epochs_monitor_test_loss, label='Test Loss', color='orange')
    plt.axvline(x=target_epoch, color='red', linestyle='--', label=f'Target Epoch {target_epoch}')
    plt.axhline(y=target_train_loss, color='blue', linestyle='--', label=f'Target Train Loss {target_train_loss:.4f} ± {target_train_loss_err:.4f}')
    plt.axhline(y=target_test_loss, color='orange', linestyle='--', label=f'Target Test Loss {target_test_loss:.4f} ± {target_test_loss_err:.4f}')
    # log scale
    # plt.yscale('log')
    plt.legend(loc = 'upper center')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.tight_layout()

    # Plotting the accuracy over epochs
    plt.subplot(1, 2, 2)
    epochs_error_rate_test = 1 - epochs_monitor_test_accuracy
    epochs_error_rate_train = 1 - epochs_monitor_accuracy
    # plt.plot(epochs_monitor_accuracy, label='Training Accuracy', color='green')
    # plt.plot(epochs_monitor_test_accuracy, label='Test Accuracy', color='red')
    plt.plot(epochs_error_rate_train, label='Training Error Rate', color='green')
    plt.plot(epochs_error_rate_test, label='Test Error Rate', color='red')
    plt.axvline(x=target_epoch, color='red', linestyle='--', label=f'Target Epoch {target_epoch}')

    target_train_accuracy = 1 - target_train_accuracy
    target_test_accuracy = 1 - target_test_accuracy
    plt.axhline(y=target_train_accuracy, color='green', linestyle='--', label=f'Target Train Error {target_train_error_rate*100:.4f}% ± {target_train_error_rate_err*100:.4f}%')
    plt.axhline(y=target_test_accuracy, color='red', linestyle='--', label=f'Target Test Error {target_test_error_rate*100:.4f}% ± {target_test_error_rate_err*100:.4f}%')
    plt.title('Error Rate over Epochs')
    plt.ylim(1e-4, 1.0)
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.legend(loc='upper center')

    # add notation with total neurons and parameters on bottom left corner
    plt.annotate(f'Total Neurons: {total_neurons}\nTotal Parameters: {total_parameters}', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=13, color='black', ha='left', va='bottom')
    plt.tight_layout()

    plt_name = 'dump/res_' + time.strftime("%Y%m%d-%H%M%S") + ".png"
    plt.savefig(plt_name)

    print(f"Plots saved to {plt_name}")

    # Save the parameters to a JSON file
    params = {
        'w1': w1.tolist(),
        'b1': b1.tolist(),
        'w2': w2.tolist(),
        'b2': b2.tolist(),
        'w3': w3.tolist(),
        'b3': b3.tolist(),
        'w4': w4.tolist(),
        'b4': b4.tolist(),
        'w1b': w1b.tolist(),
        'b1b': b1b.tolist(),
        'w2b': w2b.tolist(),
        'b2b': b2b.tolist(),
        'total_neurons': total_neurons,
        'total_parameters': total_parameters,
        'epochs_monitor_loss': epochs_monitor_loss.tolist(),
        'epochs_monitor_test_loss': epochs_monitor_test_loss.tolist(),
        'epochs_monitor_accuracy': epochs_monitor_accuracy.tolist(),
        'epochs_monitor_test_accuracy': epochs_monitor_test_accuracy.tolist()
    }
    params_file = 'dump/params_' + time.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {params_file}")