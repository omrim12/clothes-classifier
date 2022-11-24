import os
import CONSTANTS
import CLI_tools as cli
import CNN_tools as cnnt
from keras.models import save_model, load_model


def main():
    # Initialize CNN model if already trained
    if os.path.exists(CONSTANTS.fashion_model_file):
        fashion_model = load_model(CONSTANTS.fashion_model_file)
    else:
        # Train a new CNN model
        fashion_model = cnnt.CNN_train()

        # Save CNN model locally
        save_model(
            fashion_model,
            CONSTANTS.fashion_model_file,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True
        )

    # Run a CLI session based on the trained model
    cli.CLI_session(cnn_model=fashion_model)


if __name__ == '__main__':
    main()
