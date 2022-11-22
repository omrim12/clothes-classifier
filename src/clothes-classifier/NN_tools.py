from tensorflow import keras
from tensorflow.keras import layers, regularizers


def test_results(X_test, y_test, model):
    """
    :param X_test: Testing dataset
    :param y_test: Testing labels
    :param model: NN / CNN trained model.
    :return: Accuracy & loss rate

    This function tests performance of pre-trained
    NN/CNN model with given test set and their
    corresponding labels.
    """
    # Computing average accuracy over the entire test set
    test_loss, test_acc = model.evaluate(X_test, y_test)

    return test_loss, test_acc


def init_network(nn_arch, regularization=True):
    """
    :param regularization: Regularization usage indicator (preventing overfitting)
    :param nn_arch: Neural Network architecture defined by a matrix of shape HIDDEN_LAYERSx2, where
                    each entry defines number of units to be initiated in a hidden layer
                    and the activation func applied within the layer
                    (#nn_arch_rows = number of hidden layers)
    :return: Neural Network model
    """
    # Topology & activation definition
    if regularization:
        model = keras.Sequential([layers.Dense(nn_arch[i][0], activation=nn_arch[i][1],
                                               kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
                                  for i in range(len(nn_arch))])
    else:
        model = keras.Sequential([layers.Dense(nn_arch[i][0], activation=nn_arch[i][1])
                                  for i in range(len(nn_arch))])

    # Compilation step
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def NN_simulation(X_train, y_train, X_test, y_test, model, epoch=5, learning_rate=0.01):
    """
    :param y_test: Testing dataset labels
    :param X_test: Testing dataset
    :param epoch: Number of learning iterations on ANN
    :param learning_rate: Learning rate hyper param
    :param X_train: Training dataset
    :param y_train: Training dataset labels
    :param model: NN network model
    :return: NN trained model.

    This function runs a simulation of NN
    learning algorithm using tensorflow library.
    """
    # Setting learning rate
    keras.backend.set_value(model.optimizer.learning_rate, learning_rate)

    # Running learning simulation
    model.fit(X_train, y_train, epochs=epoch, batch_size=128)

    # Analyze recognition success & loss rate
    # upon testing dataset
    loss, acc = model.evaluate(X_test, y_test)

    return model, loss, acc
