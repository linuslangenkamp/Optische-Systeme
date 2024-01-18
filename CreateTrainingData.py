"""
Last Edit 18.01.2024
@author: Linus Langenkamp
"""

from vmbpy import *
import numpy as np
import cv2
import time
import Util

print("Packages imported...")
#%% create training data

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
            # extraction, adding to tempList
            else:
                print(it)
                extracted = Util.extractBG(frame, frameBG, kernelSmall, kernelBig)
                extracted = cv2.resize(extracted, (200, 200))
                if 1201 > it > 200:
                    tempList.append(extracted)
                if it > 1100:
                    break
                cv2.imshow('frame', extracted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            it += 1
print("Took {} seconds to record".format(time.time() - timeStart))

#%% export training data
tempPath = r'C:\Users\Linus\Desktop\Studium\Master\Optische Systeme\Projekt\archive\asl_alphabet_train_noBG\temp'
letter = "Z"
letterNumber = 901
for image in tempList:
    cv2.imwrite(tempPath + "\\" + letter + str(letterNumber) + ".jpg", image)
    letterNumber += 1
