from termcolor import cprint
from img_utils import analyze_img
from CNN_tools import classify_client_input


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
                # Convert given image to numpy array
                image_array = analyze_img(command[1])
                if image_array is not None:
                    cprint(f"Your image has been classified as {classify_client_input(image_array, cnn_model)}!\n",
                           "green")
                else:
                    cprint(f"Invalid path to image given. Please try again\n", "red")

        elif command[0] == 'exit':
            if len(command) > 1:
                cprint("Invalid command. Please try again\n", "red")
            else:
                exit_cli = True

