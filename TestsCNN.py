"""
Last Edit 21.01.2024
@author: Linus Langenkamp
"""

from vmbpy import *
import time
import numpy as np
import cv2
import skimage
from skimage.transform import resize
from tensorflow import keras
import Util

print("Packages imported...")
#%% import model

model = keras.models.load_model('models\CNN_NoBG_ext_TO.h5')
#%% define constants

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
#%% real time analysis

with VmbSystem.get_instance() as vmb:
    timeStart = time.time()
    tempList = []
    it = 0
    write = False
    lastframes = []
    bglist = []
    kernelSmall = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    kernelBig = kernel = np.ones((8, 8), dtype=np.uint8)

    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        handler = Util.Handler()
        cam.start_streaming(handler=handler, buffer_count=10)
        while True:
            frame = handler.get_image()
            # reference background
            if it < 24:
                bglist.append(frame)
            elif it == 24:
                frameBG = np.mean(bglist, axis=0)
            # extraction, resizing, evaluation, best match
            else:
                extracted = Util.extractBG(frame, frameBG, kernelSmall, kernelBig)
                extracted = cv2.resize(extracted, (200, 200))
                if it >= 100:
                    imgResized = skimage.transform.resize(extracted, (imageSize, imageSize, 3))
                    img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
                    evaluation = model(img)
                    argmax = np.argmax(evaluation)
                    print("%s, %s" % (Util.inverse(argmax), evaluation[0, argmax].numpy()))
                cv2.imshow('frame', extracted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            it += 1
print("Total running time: {} seconds".format(time.time() - timeStart))
