import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import image processing libraries
import cv2
import skimage
from skimage.transform import resize

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os

print("Packages imported...")

x = np.array([[1., 2., 3.],
            [4., 5., 6.],
              [7., 8., 9.]])
x = np.reshape(x, [1, 3, 3, 1])
max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding="valid")

y = max_pool_2d(x)
print(y)
