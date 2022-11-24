import numpy as np
from PIL import Image
from pathlib import Path
import CNN_tools as cnnt
from termcolor import cprint
import matplotlib.pyplot as plt


def image_2_np(image_name: str) -> np.array:
    """
    image_path: client given image to classify
    return: image converted to np array if exists else None

    This function parses actual image, normalizes and flattens
    it within a numpy array to be given as an input to classify
    using a trained CNN model.

    *Note: Image should be mounted under project directory.*
    """
    try:
        # Try opening the image and resize. if missing, return None.
        img = Image.open(image_name).resize((28, 28), Image.ANTIALIAS)
        plt.imshow(img)
        plt.show()
        exit(1)

        # Convert image to numpy array and return it.
        return np.array(5)

    except FileNotFoundError as fnf:
        return None


def usage_prompt():
    return ("\n~~~~~~ CLI usage options: ~~~~~~\n"
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
                # Convert given image to np array
                image_array = image_2_np(command[1])
                if image_array:
                    cprint(f"Your image has been classified as {cnnt.classify_input(image_array, cnn_model)}!\n",
                           "green")
                else:
                    cprint(f"Invalid path to image given. Please try again\n", "red")

        elif command[0] == 'exit':
            if len(command) > 1:
                cprint("Invalid command. Please try again\n", "red")
            else:
                exit_cli = True
                continue
