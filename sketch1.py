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

# 60k training samples, 10k test samples
# read idx1-ubyte and idx3-ubyte files

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

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()
    
    logger.debug(f"Training data shape: {train_data.shape}")
    logger.debug(f"Training labels shape: {train_labels.shape}")
    logger.debug(f"Test data shape: {test_data.shape}")
    logger.debug(f"Test labels shape: {test_labels.shape}")

    number_of_plots = 10
    
    # Show the first 10 training images
    plt.figure(figsize=(10, 4))
    for i in range(number_of_plots):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_data[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {train_labels[i][0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()