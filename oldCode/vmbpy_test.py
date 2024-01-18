# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:05:40 2024

@author: Linus
"""

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
#%%
# load trained model
model = keras.models.load_model('asl_cnn.h5')
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
#%%
def at(x, index):
    return np.array([x[index], ])


def inverse(index):
    if 0 <= index < 26:
        label = chr(ord("A") + index)
    elif index == 26:
        label = 'del'
    elif index == 27:
        label = 'nothing'
    elif index == 28:
        label = 'space'
    else:
        label = 'error'

    return label


def allEvaluations(tensor):
    for index in range(26):
        print("%s: %s" % (chr(ord("A") + index), tensor[0,index].numpy()))
    print("del: %s" % (tensor[0,26].numpy()))
    print("nothing: %s" % (tensor[0,27].numpy()))
    print("space: %s" % (tensor[0,28].numpy()))
    print("\nbest letter: %s" % inverse(np.argmax(tensor)))
    print("probability: %s" % (np.max(tensor)))
    return


def best(tensor):
    print("\nbest letter: %s" % inverse(np.argmax(tensor)))
    print("probability: %s" % (np.max(tensor)))
    return


def diffBG(bg, tensor):
    diff = np.abs(tensor - bg)

#%% capture
from vmbpy import *
import cv2
