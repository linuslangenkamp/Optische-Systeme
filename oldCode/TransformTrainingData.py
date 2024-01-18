# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:05:40 2024

@author: Linus
"""
from queue import Queue

from vmbpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
# import image processing libraries
import cv2
import skimage
from skimage.transform import resize
from numba import jit
# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os
import BGExtractor

print("Packages imported...")
#%%
tstart = time.time()
batch_size = 64
imageSize = 200
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = '../archive/asl_alphabet_train/asl_alphabet_train/'


def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            for image_filename in os.listdir(folder + folderName):
                X[cnt] = cv2.imread(folder + folderName + '/' + image_filename)
                cnt += 1
    return X

X_train= get_data(train_dir)
print("Images successfully imported...")
print("Took %s seconds to import" % (time.time() - tstart))
#%%
def extractBG(frame, bg, bound=0.08):
    mask = abs(bound * (frame - bg)).astype(np.uint8)
    mask[mask >= 1] = 1
    mask[np.any(mask, axis=2)] = 1
    extracted = (mask * frame).astype(np.uint8)
    return extracted

@jit(nopython=True)
def diffCornerData(img, reference):
    difference = 0
    for i in range(199):
        for k in range(3):
            difference += abs(img[8][i][k] - reference[8][i][k])
            difference += abs(img[192][i][k] - reference[192][i][k])
            difference += abs(img[i][8][k] - reference[i][8][k])
            difference += abs(img[i][192][k] - reference[i][192][k])
    return difference

def bestBGIdx(img, nothing):
    L = []
    for n in nothing:
        L.append(diffCornerData(img, n))
    return np.argmin(L)
#%%
# nothing 45000ff
nothing = X_train[45000:48000]
bestNothing = dict()

#%%
for idx, img in enumerate(X_train):
    bestBG = bestBGIdx(img, nothing)
    print(bestBG)
    bestNothing[idx] = bestBGIdx(img, nothing)
    extracted = extractBG(img, nothing[bestBG], bound=0.04)
    cv2.imshow('extracted', extracted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

