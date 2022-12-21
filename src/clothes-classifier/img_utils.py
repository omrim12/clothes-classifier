import cv2
import numpy as np
from rembg import remove
from CONSTANTS import image_size


def analyze_img(image_name: str) -> np.array:
    """
    image_path: client given image to classify
    return: image converted to np array if exists else None

    This function parses actual image:
    it performs bacokground removal, normalizes and flattens
    it within a numpy array to be given as an input to classify
    using a trained CNN model.

    *Note: Image should be mounted under project directory.*
    """
    try:
        # Applying background removal using openCV library tools
        img_origin = cv2.imread(image_name)
        img_origin = cv2.resize(img_origin, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img_clean = remove(img_origin)

        # Convert image to numpy array with RGB average values
        # and scaled pixel intensities in range 0-1
        img_array = np.mean(np.array(img_clean), axis=2) / 255.
        
        return img_array

    # In case of missing input file
    except cv2.error:
        return None