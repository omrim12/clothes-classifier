import os
import numpy as np
import CNN_tools as cnnt
from termcolor import cprint


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

    while not exit_cli:
        # Show usage prompt
        cprint(usage_prompt(), "yellow")

        # Get client input
        command = input("#> ").split(" ")

        # Classifier option
        if command[0] == 'classify':
            if len(command) != 2:
                cprint("Invalid command. Please try again\n", "red")
            else:
                image_array = image_2_np(command[1])
                if image_array:
                    cprint(f"Your image has been classified as {cnnt.classify_input(image_array, cnn_model)}!\n",
                           "green")
                else:
                    print(f"Invalid path to file given. Please try again\n", "red")

        elif command[0] == 'exit':
            if len(command) > 1:
                cprint("Invalid command. Please try again\n", "red")
            else:
                exit_cli = True
                continue
