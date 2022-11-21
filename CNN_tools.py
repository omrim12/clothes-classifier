import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


def ReLU(Z):
    """
    :return: ReLU transform of Z
    """
    A = np.copy(Z)
    A[A < 0] = 0
    return A


def zero_pad(X, p):
    """
    zero_pad -zero padding all feature maps in a dataset X.
    The padding is carried out on the height and width dimensions.
    Input Arguments:
    X - np array of shape (m, height, width, n_c)
    p - integer, number of zeros to add on each side
    Returns:
    Xp - X after zero padding of size (m, height + 2 * p, width + 2 * p, n_c)
    """
    return np.pad(X, [(0, 0), (p, p), (p, p), (0, 0)], mode='constant', constant_values=(0, 0))


def conv(fmap_patch, filtMat, b):
    """
     conv - apply a dot product of one patch of a previous layer feature map
     Input Arguments:
     fmap_patch - patch of the input data of shape (f, f, n_c)
     filtMat - Weight parameters of shape (f, f, n_c)
     b - Bias parameters of shape (1, 1, 1)
     Returns:
     y - a scalar value, the result of convulsing the sliding window (W, b) on a slice of the input data
     """
    return np.sum([np.sum(fmap_patch[:, :, i] * filtMat[:, :, i])
                   for i in range(filtMat.shape[2])]) + b


def load_clothes_data(data_order='cnn'):
    """
    return: Clothes dataset assembled of both fashion MNIST
            and glasses datasets
    """
    # Loading fashion MNIST datasets
    X_train_mnist, y_train_mnist, \
    X_valid_mnist, y_valid_mnist, \
    X_test_mnist, y_test_mnist = load_fashion_mnist_data(data_order=data_order)

    # Loading glasses datasets
    X_train_glass, y_train_glass,\
    X_valid_glass, y_valid_glass,\
    X_test_glass, y_test_glass = load_glasses_data(data_order=data_order)

    # Merging datasets
    # TODO: implement this block

    return None, None, None, None, None, None


def load_glasses_data(data_order='cnn'):
    """
    return: Glasses datasets seperated to train,
            validation and test datasets
    """
    # TODO: implement this function using appropriate database
    return None, None, None, None, None, None


def load_fashion_mnist_data(data_order='cnn'):
    """
    :return: Fashion MNIST dataset seperated to train,
             validation and test datasets.
    """
    # Initializing fashion MNIST data from keras dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
    X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
    X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

    # scale the pixel intensities down to 0-1 range by dividing them by 255.0
    # this also converts them to floats.
    X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

    # Convert dataset tensor to a 2D matrix
    if data_order == 'fcn':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] ** 2))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1] ** 2))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] ** 2))

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def conv_forward(Fmap_input, filt_weights, b, p=0, s=1):
    """
     Forward propagation - convnet
     Input Arguments:
     Fmap_input - input feature maps (or output of previous layer),
     np array (m, n_H, n_W, n_C)
     m - number of input samples, n_H - hight, n_W - 'width', n_C - number of channels
     filt_weights - Weights, numpy array of shape (f, f, n_C, n_filt)
     b - bias, numpy array of shape (1, 1, 1, n_filt)
     p - padding parameter (default: p = 0), s - stride (default: s = 1)
     Returns:
     Fmap_output - output, numpy array of shape (m, n_H, n_W, n_filt)
    """
    # Initializing output map
    out_height = int(np.floor(((Fmap_input.shape[1] - filt_weights.shape[0] + 2 * p) / s) + 1))
    out_width = int(np.floor(((Fmap_input.shape[2] - filt_weights.shape[1] + 2 * p) / s) + 1))
    num_samples = Fmap_input.shape[0]
    num_filters = filt_weights.shape[3]
    output_map = np.zeros(shape=(num_samples, out_width, out_height, num_filters))

    # Performing convolution process:
    # Convulsing input feature map patches with each filter
    f = filt_weights.shape[0]
    n_w = Fmap_input.shape[2]
    n_h = Fmap_input.shape[1]

    for m in range(num_samples):
        for i in range(0, n_h - f + 1, s):
            for j in range(0, n_w - f + 1, s):
                for k in range(num_filters):
                    # Extracting current stride's patch to convolve with filters
                    curr_patch = Fmap_input[m, i:i + f, j:j + f, :]
                    curr_filter = filt_weights[:, :, :, k]
                    curr_bias = b[:, :, :, k]

                    # Convulsing filter and patch and applying ReLU activation
                    conv_res = ReLU(conv(curr_patch, curr_filter, curr_bias))

                    # Applying results at output map
                    output_map[m, int(i / s), int(j / s), k] = conv_res[0, 0, 0]

    # Zero padding output map
    output_map = zero_pad(output_map, p)

    return output_map


def CNN_simulation(X_train, y_train, X_valid, y_valid, X_test, y_test, conv_type='valid'):
    """
    :param conv_type: Convolution padding method (valid/same)
    :param X_train: Training dataset
    :param y_train: Training labels
    :param X_valid: Validation dataset
    :param y_valid: Validation labels
    :param X_test: Testing dataset
    :param y_test: Testing labels
    :return: CNN trained model & accuracy rate

    This function runs a CNN learning simulation
    based on fixed parameters that define the convolution network topology.
    It performs feature extraction using convolution & max-pooling layers
    and eventually runs a NN learning process to produce a model corresponding
    to the fashion MNIST dataset.
    """
    # Init CNN inputs tensor
    inputs = keras.Input(shape=(28, 28, 1))

    # Run convolution & max-pooling
    X = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding=conv_type)(inputs)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding=conv_type)(X)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding=conv_type)(X)

    # Flatten output layer (a KerasTensor) from CNN to pass output data
    # to NN output dense layer (knows only to receive a vector)
    X = layers.Flatten()(X)

    # Calculating outputs from NN dense layer
    outputs = layers.Dense(10, activation='softmax')(X)

    # Initialize model derived from CNN results
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Summarize results from CNN
    model.summary()

    # Running NN simulation on trained model derived from CNN
    print("\nRunning NN learning process...\n")
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64)

    # Computing loss & accuracy over the entire test set
    test_loss, test_acc = model.evaluate(X_test, y_test)

    return model, test_loss, test_acc
