import numpy as np
import pandas as pd
import cv2 as cv
import PIL.Image
from IPython.display import display
import tensorflow as tf
import random
import matplotlib.pyplot as plt

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

plot_model = lambda m: tf.keras.utils.plot_model(m,to_file="./ignore/model.png",show_shapes=True, show_layer_names=False)

def plot_history(history):
    # Plot learning curves
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('MSE curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='lower right')

    plt.tight_layout()
    plt.show()

def save_model(model, json_filename, weights_filename):
    model_json = model.to_json()
    with open(json_filename, 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(weights_filename)



def load_model(json_filename, weights_filename):
    with open(json_filename, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = models.model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_filename)
    return loaded_model


def get_trainable_variables(model):
	return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def save_history_to_pd(history,output):
	hist_df = pd.DataFrame(history.history) 
	hist_df.to_csv(output)