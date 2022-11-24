import CNN_tools as cnnt
import CLI_tools as cli


def main():
    # Initialize and train a CNN model
    fashion_model = cnnt.CNN_train()

    # Run a CLI session using the trained model
    cli.CLI_session(cnn_model=fashion_model)


if __name__ == '__main__':
    main()
