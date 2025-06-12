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

local_block_size_l1 = 3 # unit of pixels
local_block_step_l1 = 1 # unit of pixels
local_block_size_l2 = 2 # unit of local blocks on layer 1
local_block_step_l2 = 2 # unit of local blocks on layer 1

def init_NN_parameters_weighted_local():
    img_size     = 28
    layer_1_size = ((img_size - local_block_size_l1) // local_block_step_l1 + 1) ** 2
    layer_2_size = ((int(np.sqrt(layer_1_size)) - local_block_size_l2) // local_block_step_l2 + 1) ** 2
    layer_3_size = 25
    layer_4_size = 10

    print(" --- Neural Network Parameters -------------------------")
    print(f"Layer 1 size: {layer_1_size}, Layer 2 size: {layer_2_size}")
    print(f"Layer 3 size: {layer_3_size}, Layer 4 size: {layer_4_size}")
    total_neurons = layer_1_size + layer_2_size + layer_3_size + layer_4_size
    print(f"Total neurons: {total_neurons}")
    total_parameters = (layer_1_size * img_size * img_size) + (layer_2_size * layer_1_size) + (layer_3_size * layer_2_size) + (layer_4_size * layer_3_size)
    print(f"Total parameters: {total_parameters}")
    print("--------------------------------------------------------")

    sigma = 3

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

    return w1, b1, w2, b2, w3, b3, w4, b4, total_neurons, total_parameters

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

def forward_propagation(X, w1, b1, w2, b2, w3, b3, w4, b4):
    leaky_rate = 1e-2
    z1 = w1.dot(X) + b1
    a1 = leaky_ReLu(z1, alpha=leaky_rate)
    z2 = w2.dot(a1) + b2
    a2 = leaky_ReLu(z2, alpha=leaky_rate)
    z3 = w3.dot(a2) + b3
    a3 = leaky_ReLu(z3, alpha=leaky_rate)

    a3_drop, dropout_mask = dropout(a3, p=0.1)
    z4 = w4.dot(a3_drop) + b4

    softmax_temp = 3.0
    a4 = softmax(z4, softmax_temp)

    return z1, a1, z2, a2, z3, a3, z4, a4, dropout_mask, softmax_temp

def back_propagation_momentum(X, Y, z1, a1, z2, a2, z3, a3, z4, a4, w1, w2, w3, w4, dw1_last=None, db1_last=None, dw2_last=None, db2_last=None, dw3_last=None, db3_last=None, dw4_last=None, db4_last=None, beta=0.90, dropout_mask=None, softmax_temp=2.0):
    m = Y.shape[1]
    dZ4 = a4 - Y
    if softmax_temp != 1.0:
        dZ4 *= softmax_temp
    dw4 = dZ4.dot(a3.T) / m
    db4 = np.sum(dZ4, axis=1, keepdims=True) / m

    dZ3_raw = w4.T.dot(dZ4) * der_leaky_ReLu(z3)
    if dropout_mask is not None:
        dZ3_raw *= dropout_mask

    dZ3 = dZ3_raw

    dw3 = dZ3.dot(a2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2_raw = w3.T.dot(dZ3) * der_leaky_ReLu(z2)
    dZ2 = dZ2_raw

    dw2 = dZ2.dot(a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1_raw = w2.T.dot(dZ2) * der_leaky_ReLu(z1)
    dZ1 = dZ1_raw

    dw1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    dw1 = mom(dw1, dw1_last)
    db1 = mom(db1, db1_last)
    dw2 = mom(dw2, dw2_last)
    db2 = mom(db2, db2_last)
    dw3 = mom(dw3, dw3_last)
    db3 = mom(db3, db3_last)
    dw4 = mom(dw4, dw4_last)
    db4 = mom(db4, db4_last)

    return dw1, db1, dw2, db2, dw3, db3, dw4, db4

# * === Loss Function ===================================================================
# * =====================================================================================
def loss_function(Y, Y_hat):
    # return loss_func_categorical_crossentropy(Y, Y_hat)
    return loss_func_squared(Y, Y_hat)
    # return loss_func_log(Y, Y_hat)


def zoom_image(image, zoom_x=1.0, zoom_y=1.0):
    h, w = image.shape
    new_h, new_w = int(h * zoom_y), int(w * zoom_x)
    
    zoomed_image = np.zeros((new_h, new_w), dtype=image.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            orig_i = int(i / zoom_y)
            orig_j = int(j / zoom_x)
            if orig_i < h and orig_j < w:
                zoomed_image[i, j] = image[orig_i, orig_j]
    
    return zoomed_image

def rotate_image(image, angle_deg=0):
    angle_rad   = np.deg2rad(angle_deg)
    cos_a       = np.cos(angle_rad)
    sin_a       = np.sin(angle_rad)

    h, w        = image.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    y, x    = np.indices((h, w))
    x_flat  = x.flatten() - cx
    y_flat  = y.flatten() - cy

    x_rot = cos_a * x_flat + sin_a * y_flat + cx
    y_rot = -sin_a * x_flat + cos_a * y_flat + cy

    x_rot = np.round(x_rot).astype(int)
    y_rot = np.round(y_rot).astype(int)

    mask = (x_rot >= 0) & (x_rot < w) & (y_rot >= 0) & (y_rot < h)
    rotated_image = np.zeros_like(image)
    rotated_image[y.flatten()[mask], x.flatten()[mask]] = image[y_rot[mask], x_rot[mask]]
    return rotated_image

# * === Main Execution ==================================================================
# * =====================================================================================
if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()

    # rotate the training data
    number_of_rotations = 4
    rotation_limit = 20
    # create number_of_rotations copies of the training data, each rotated by a random angle
    train_data_rotated = []
    for i in range(number_of_rotations):
        angle = i * 2 * rotation_limit / (number_of_rotations - 1) - rotation_limit
        # ramdom x and y zoom factor between 0.8 and 1.2
        zoom_factor_x = np.random.uniform(0.8, 1.2)
        zoom_factor_y = np.random.uniform(0.8, 1.2)
        print(f"Rotating training data by {angle:.2f} degrees...")
        rotated_data = np.array([rotate_image(image, angle) for image in train_data])
        train_data_rotated.append(rotated_data)

    # print the first 5 images with all rotations
    # for i in range(5):
    #     plt.figure(figsize=(10, 2))
    #     for j, rotated_data in enumerate(train_data_rotated):
    #         plt.subplot(1, number_of_rotations + 1, j + 1)
    #         plt.imshow(rotated_data[i], cmap='gray')
    #         plt.title(f'Rotation {j * 60 - 30}°')
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'dump/rotated_image_{i}.png')

    # print("Smoothing training data with a 3x3 kernel...")
    # train_smoothed_data = image_LPF(train_data, kernel_size=3)
    
    # print(f"Training data shape: {train_data.shape}")
    # print(f"Training labels shape: {train_labels.shape}")
    # print(f"Test data shape: {test_data.shape}")
    # print(f"Test labels shape: {test_labels.shape}")

    number_of_plots         = 10
    target_epoch            = 1500
    epoch_print_every       = 10
    target_train_accuracy   = 0.0
    target_test_accuracy    = 0.0
    target_train_loss       = 0.0
    target_test_loss        = 0.0

    epochs            = 1600
    learning_rate_max = 0.05
    learning_rate_min = 0.01
    batch_size        = 256
    momentum_beta     = 0.9

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
    
    train_labels_oh = one_hot(train_labels)  # (10,60000)
    test_labels_oh  = one_hot(test_labels)   # (10,10000)

    # Split data into batches
    train_data_batches  = np.array_split(train_data,  train_data.shape[1]//batch_size, axis=1)
    train_labels_batches= np.array_split(train_labels_oh, train_labels_oh.shape[1]//batch_size, axis=1)
    train_rotated_data_batches = [np.array_split(rotated_data, rotated_data.shape[1]//batch_size, axis=1) for rotated_data in train_rotated_data]
    # train_smoothed_data_batches = np.array_split(train_smoothed_data, train_smoothed_data.shape[1]//batch_size, axis=1)

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

    w1, b1, w2, b2, w3, b3, w4, b4, total_neurons, total_parameters = init_NN_parameters_weighted_local()
    dw1_last = np.zeros_like(w1)
    db1_last = np.zeros_like(b1)
    dw2_last = np.zeros_like(w2)
    db2_last = np.zeros_like(b2)
    dw3_last = np.zeros_like(w3)
    db3_last = np.zeros_like(b3)
    dw4_last = np.zeros_like(w4)
    db4_last = np.zeros_like(b4)

    # make the progress bar float
    for epoch in range(epochs):

        learning_rate = linear_learning_rate(epoch, epochs, initial_lr=learning_rate_max, final_lr= learning_rate_min)

        z1_epoch, a1_epoch, z2_epoch, a2_epoch, z3_epoch, a3_epoch, z4_epoch, a4_epoch, mask_dump, softmax_temp = forward_propagation(train_data, w1, b1, w2, b2, w3, b3, w4, b4)

        z1_test_epoch, a1_test_epoch, z2_test_epoch, a2_test_epoch, z3_test_epoch, a3_test_epoch, z4_test_epoch, a4_test_epoch, make_dump, softmax_temp_test = forward_propagation(test_data, w1, b1, w2, b2, w3, b3, w4, b4)

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

        # if epoch == target_epoch:
        #     # target_train_accuracy = accuracy_train
        #     # target_test_accuracy = accuracy_test
        #     target_train_loss = loss
        #     target_test_loss = loss_test

        # print the loss and accuracy every 100 epochs
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

            z1, a1, z2, a2, z3, a3, z4, a4, mask, softmax_temp = forward_propagation(train_batch, w1, b1, w2, b2, w3, b3, w4, b4)

            loss = loss_function(train_labels_batch, a4)

            dw1, db1, dw2, db2, dw3, db3, dw4, db4 = back_propagation_momentum(train_batch, train_labels_batch, z1, a1, z2, a2, z3, a3, z4, a4, w1, w2, w3, w4, dw1_last=dw1_last, db1_last=db1_last, dw2_last=dw2_last, db2_last=db2_last, dw3_last=dw3_last, db3_last=db3_last, dw4_last=dw4_last, db4_last=db4_last, beta=momentum_beta, dropout_mask=mask, softmax_temp=softmax_temp)

            dw1_last, db1_last = dw1, db1
            dw2_last, db2_last = dw2, db2
            dw3_last, db3_last = dw3, db3
            dw4_last, db4_last = dw4, db4

            w1, b1, w2, b2, w3, b3, w4, b4 = update_NN_parameters_limited(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, learning_rate)

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
    plt.ylim(1e-3, 1.0)
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