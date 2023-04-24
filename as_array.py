import numpy as np
import os
import random
from PIL import Image


# Libraries for TensorFlow
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow import keras

# Library for Transfer Learning
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array, array_to_img
from tensorflow.keras.utils import load_img



def as_array (x):
  y = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x])
  return y
