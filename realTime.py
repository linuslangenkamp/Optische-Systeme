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

# define a video capture object
vid = cv2.VideoCapture(0)
for j in range(50):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
vid.set(cv2.CAP_PROP_EXPOSURE, -4.678071905)
exp = vid.get(cv2.CAP_PROP_EXPOSURE)
it = 0
lastframes = []
bglist = []
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if it < 24:
        bglist.append(frame)
    if it == 24:
        frameBG = np.mean(bglist, axis=0) #.astype(np.uint8)
    if len(lastframes) < 5:
        lastframes.append(frame)
    if len(lastframes) == 5 and it > 24:
        lastframes.pop(0)
        lastframes.append(frame)
        avg = np.mean(lastframes, axis=0)
        f = abs(0.1 * (avg - frameBG)).astype(np.uint8)
        f[f >= 1] = 1
        f[np.any(f, axis=2)] = 1
        f = f * lastframes[0]
        # Display the resulting frame
        cv2.imshow('frame', frame)

        frameResized = skimage.transform.resize(f, (imageSize, imageSize, 3))
        img = np.asarray(frameResized).reshape((-1, imageSize, imageSize, 3))
        evaluation = model(img)
        best(evaluation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    it += 1
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()