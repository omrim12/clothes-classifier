import numpy as np
import tensorflow as tf
from tabulate import tabulate
import tensorflow.keras as keras
from tensorflow.keras import layers


def load_clothes_data():
    """
    return: Clothes dataset assembled of both fashion MNIST
            and glasses datasets
    """
    # Loading fashion MNIST datasets
    X_train_mnist, y_train_mnist, \
    X_valid_mnist, y_valid_mnist, \
    X_test_mnist, y_test_mnist = load_fashion_mnist_data()

    # Loading glasses datasets
    X_train_glass, y_train_glass, \
    X_valid_glass, y_valid_glass, \
    X_test_glass, y_test_glass = load_glasses_data()

    # Merging datasets
    # TODO: implement this block

    return None, None, None, None, None, None


def load_glasses_data():
    """
    return: Glasses datasets seperated to train,
            validation and test datasets
    """
    # TODO: implement this function using appropriate database
    return None, None, None, None, None, None


def load_fashion_mnist_data():
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

    return X_train, y_train, X_valid, y_valid, X_test, y_test


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


def classify_input(image_array: np.array, cnn_model) -> str:
    """
    image_array: input image converted to numpy array
    return: clothing item type

    This function recieves an image represented by numpy array
    and returns most similar clothing item determined by patterns
    that were trained by CNN model
    """
    # TODO: Adjust image size

    # TODO: pass image in cnn_model to get compatible clothing item

    # TODO: classify result-to-cloth name (might need to use enum/dict)

    return "CLOTH"


def CNN_train():
    """
    returns trained CNN model based on fashion MNIST database.
    """
    # TODO: Should use load_clothes_data after implementing glasses and MNIST datasets merge
    # Loading fashion MNIST datasets for CNN input
    X_train_cnn, y_train_cnn, X_valid_cnn, y_valid_cnn, X_test_cnn, y_test_cnn = load_fashion_mnist_data()

    # a. Training CNN "same" using the fashion MNIST dataset
    print("--- Running CNN 'same' learning simulation using the fashion MNIST dataset ---\n")
    same_model, same_loss, same_acc = CNN_simulation(X_train_cnn, y_train_cnn,
                                                     X_valid_cnn, y_valid_cnn,
                                                     X_test_cnn, y_test_cnn,
                                                     conv_type='same')

    # Training CNN "valid" using the fashion MNIST dataset
    print("\n\n--- Running CNN 'valid' learning simulation using the fashion MNIST dataset ---\n")
    valid_model, valid_loss, valid_acc = CNN_simulation(X_train_cnn, y_train_cnn,
                                                        X_valid_cnn, y_valid_cnn,
                                                        X_test_cnn, y_test_cnn,
                                                        conv_type='valid')

    # Comparing results of 'same' & 'valid' CNNs
    table = [['Accuracy', same_acc, valid_acc],
             ['Loss', same_loss, valid_loss]]
    print('\n\n', tabulate(table, headers=["Property", "'same' CNN", "'valid' CNN"]))

    # Determine best model by model accuracy
    if valid_acc > same_acc:
        return valid_model

    return same_model
