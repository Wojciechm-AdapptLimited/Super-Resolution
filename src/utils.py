import numpy as np
import cv2 as cv
import PIL.Image
from IPython.display import display
import tensorflow as tf
import random

def imshow(img: np.ndarray):
    """
    Displays the image, courtesy of the lab notebooks

    :param img: Image to display
    """
    img = img.clip(0, 255).astype("uint8")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(img))

def scaled_imshow(img: np.ndarray, fx=0.3, fy=0.3):
    """
    Displays the image scaled by the given factor

    :param img: Image to display
    :type img: np.ndarray
    :param fx: Factor to scale the image by in the x-axis, defaults to 0.3
    :type fx: float, optional
    :param fy: Factor to scale the image by in the y-axis, defaults to 0.3
    :type fy: float, optional
    """
    scaled_img = cv.resize(img, None, fx=fx, fy=fy)
    imshow(scaled_img)

def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)