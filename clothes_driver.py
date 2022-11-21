import numpy as np
import NN_tools as nnt
import CNN_tools as cnnt
from tabulate import tabulate


def main():

    # TODO: Should use load_clothes_data after implementing glasses and MNIST datasets merge
    # Loading fashion MNIST datasets for both CNN and FCN networks
    X_train_cnn, y_train_cnn, X_valid_cnn, y_valid_cnn, X_test_cnn, y_test_cnn = cnnt.load_fashion_mnist_data()
    X_train_fcn, y_train_fcn, X_valid_fcn, y_valid_fcn, X_test_fcn, y_test_fcn = cnnt.load_fashion_mnist_data(data_order
                                                                                                              ='fcn')

    # a. Running CNN "same" learning simulation using the fashion MNIST dataset
    print("--- Running CNN 'same' learning simulation using the fashion MNIST dataset ---\n")
    same_model, same_loss, same_acc = cnnt.CNN_simulation(X_train_cnn, y_train_cnn,
                                                          X_valid_cnn, y_valid_cnn,
                                                          X_test_cnn, y_test_cnn,
                                                          conv_type='same')

    # b. Running FCN learning simulation using the fashion MNIST dataset
    print("\n\n--- Running FCN learning simulation using the fashion MNIST dataset ---'")
    # Defining network architecture
    nn_arch = np.array([[128, 'relu'],
                        [128, 'relu'],
                        [128, 'relu']])
    nn_arch = np.concatenate((nn_arch, [[len(np.unique(y_train_fcn)), 'softmax']]), axis=0)  # --> Adding output layer
    NN_model = nnt.init_network(nn_arch)
    fcn_model, fcn_loss, fcn_acc = nnt.NN_simulation(X_train_fcn, y_train_fcn,
                                                     X_test_fcn, y_test_fcn,
                                                     model=NN_model)

    # Running CNN "valid" learning simulation using the fashion MNIST dataset
    print("\n\n--- Running CNN 'valid' learning simulation using the fashion MNIST dataset ---\n")
    valid_model, valid_loss, valid_acc = cnnt.CNN_simulation(X_train_cnn, y_train_cnn,
                                                             X_valid_cnn, y_valid_cnn,
                                                             X_test_cnn, y_test_cnn,
                                                             conv_type='valid')

    # Comparing results of 'same' & 'valid' convolution networks and FCN network
    table = [['Accuracy', same_acc, valid_acc, fcn_acc],
             ['Loss', same_loss, valid_loss, fcn_loss]]
    print('\n\n', tabulate(table, headers=["Property", "'same' CNN", "'valid' CNN", "FCN"]))


if __name__ == '__main__':
    main()
