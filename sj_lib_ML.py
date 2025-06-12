import numpy as np

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels.flatten()].T

def one_hot_eplison(labels, num_classes=10, epsilon=1e-15):
    one_hot_labels = np.eye(num_classes)[labels.flatten()].T
    # replace 0s with epsilon
    one_hot_labels = np.clip(one_hot_labels, epsilon, 1 - epsilon)
    return one_hot_labels

def mom(update, last, beta=0.9):
    return beta * last + (1 - beta) * update if last is not None else update

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

def softmax(x, temp=2.0):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True)) / temp
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def der_softmax(x, temp=2.0):
    s = softmax(x, temp)
    return s * (1 - s)

def dropout(a, p):
    mask = (np.random.rand(*a.shape) > p) / (1.0 - p)
    return a * mask, mask

def accuracy(Y, Y_hat):
    correct_predictions = np.sum(np.argmax(Y, axis=0) == np.argmax(Y_hat, axis=0))
    return correct_predictions / float(Y.shape[1])

def loss_func_log(Y, Y_hat):
    epsilon = 1e-15
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=0))

def loss_func_squared(Y, Y_hat):
    return 2* np.mean(np.sum((Y - Y_hat) ** 2, axis=0))

def loss_func_categorical_crossentropy(Y, Y_hat):
    epsilon = 1e-15
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=0))

def exp_learning_rate(epoch_current, epoch_total, initial_lr=5, final_lr=0.1, decay_rate=0.1):
    if epoch_current >= epoch_total:
        return final_lr
    else:
        return max(final_lr, initial_lr * np.exp(-decay_rate * (epoch_current / epoch_total)))
    
def linear_learning_rate(epoch_current, epoch_total, initial_lr=5, final_lr=0.1):
    if epoch_current >= epoch_total:
        return final_lr
    else:
        return max(final_lr, initial_lr - (initial_lr - final_lr) * (epoch_current / epoch_total))
    
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