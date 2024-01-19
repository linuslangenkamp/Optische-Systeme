from queue import Queue
from vmbpy import *
import numpy as np
import cv2
from faker import Faker
import random
fake = Faker('de_DE')


class Handler:

    def __init__(self):
        self.display_queue = Queue(10)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # print('{} acquired {}'.format(cam, frame), flush=True)
            self.display_queue.put(frame.as_opencv_image(), True)

        cam.queue_frame(frame)


def extractBG(frame, frameBG, kernelSmall, kernelBig):
    mask = abs(0.1 * (frame - frameBG)).astype(np.uint8)
    mask[mask >= 1] = 1
    mask[np.any(mask, axis=2)] = 1
    neighbor_count = cv2.filter2D(mask, cv2.CV_8U, kernelSmall)
    mask = np.where(neighbor_count <= 3, 0, mask)
    neighbor_count = cv2.filter2D(mask, cv2.CV_8U, kernelBig)
    mask = np.where(neighbor_count <= 10, 0, mask)
    extracted = mask * frame
    return extracted


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


def shortLetter(letter):
    if letter == 'del':
        return "--"
    elif letter == 'space':
        return "_"
    elif letter == 'nothing':
        return ""
    else:
        return letter


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


def generateWord():
    w = "NULLNULL"
    while len(w) > 7 or "Ö" in w or "Ä" in w or "Ü" in w:
        w = fake.word().upper()
    return w


def generateLetter():
    return chr(65 + random.randint(0,25))
