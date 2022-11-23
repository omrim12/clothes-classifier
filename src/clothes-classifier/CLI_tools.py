import os
import numpy as np
import CNN_tools as cnnt


def image_2_np(image_path: str) -> np.array:
    """
    image_path: relative/absolute path to client given image to classify
    return: image converted to np array if path is valid else None

    This function parses a given image path to an actual image,
    normalizes and flattens it within a numpy array to be given
    as an input to classify using a trained CNN model.
    """
    # TODO: try looking opening the image. if missing, return None.

    # TODO: else, convert image to numpy array and return it.

    return np.array(5)


def usage_prompt():
    return ("~~~~~~ CLI usage options: ~~~~~~\n"
            "#> classify <cloth_image_path> - classify cloth input image\n"
            "#> exit                        - exit CLI session\n"
            "#> help / <any-other-command>  - show usage options\n")


def CLI_session(cnn_model):
    """
    cnn_model: Trained CNN model based on fashion MNIST datasets

    This function provides a client interface wrapper for classifying
    clothing items by a given image path using the trained CNN model.
    """
    exit_cli = False

    # Show usage prompt
    print(usage_prompt())

    while not exit_cli:
        # Get client input
        command = input("#> ").split(" ")

        # Classifier option
        if command[0] == 'classify':
            if len(command) != 2:
                print("Invalid command. Please try again\n")
            else:
                image_array = image_2_np(command[1])
                if image_array:
                    print(f"Your image has been classified as {cnnt.classify_input(image_array, cnn_model)}!\n")
                else:
                    print(f"Invalid path to file given. Please try again\n")

        elif command[0] == 'exit':
            if len(command) > 1:
                print("Invalid command. Please try again\n")
            else:
                exit_cli = True
                continue

        # Re-prompt usage
        print(usage_prompt())
