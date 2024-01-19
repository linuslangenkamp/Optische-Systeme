import copy

from ursina import *
import Util
import numpy as np
from PIL import Image
from tensorflow import keras
from vmbpy import *
import cv2
import skimage
from skimage.transform import resize
import copy

#%% import model

model = keras.models.load_model('models\CNN_NoBG_ext_TO.h5')
#%% define constants

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
#%%
window.title = "LiveLearner"
window.icon = r"archive\images\icon.ico"
window.borderless = True
app = Ursina()
window.color = color.rgb(55, 55, 55)
window.fps_counter.enabled = False
window.size = (1600, 900)


def buchstaben():
    mainMenu.disable()
    buchstabenMenu.enable()


def woerter():
    mainMenu.disable()
    woerterMenu.enable()


def free():
    mainMenu.disable()
    freeMenu.enable()


def backToMain():
    woerterMenu.disable()
    buchstabenMenu.disable()
    mainMenu.enable()


def beenden():
    app.destroy()
    app.userExit()


mainMenu = Entity()
mainMenu.position = (0, 0)
headline = Text(parent=mainMenu, scale=(25, 25), position=(0, 2), text="Deutsches Fingeralphabet - LiveLearner")
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Buchstaben', on_click=buchstaben, position=(-4, 0), radius=.2)
headline.setPos((-headline.width/2, 2, 0))
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Wörter', on_click=woerter, position=(0, 0), radius=.2)
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Freier Modus', on_click=woerter, position=(4, 0), radius=.2)
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Beenden', on_click=beenden, position=(0, -2), radius=.2)

woerterMenu = Entity()
woerterMenu.position = (0, 0)
Button(parent=woerterMenu, color=color.gray, scale=(3, 1), text='Back', on_click=backToMain, position=(0, -3), radius=.2)
t1 = Text(parent=woerterMenu, text='Wort: ', color=color.black, scale=(15, 15), position=(0, 2.5))
t1.setPos((-t1.width/2, 2.5, 0))
t2 = Text(parent=woerterMenu, text='Erkannt: ', color=color.black, scale=(15, 15), position=(0, 1.5))
t2.setPos((-t2.width/2, 1.5, 0))
pictogram = Entity(model='quad', scale=(3, 3, 3), position=(0, -2), parent=woerterMenu)
liveExtracted = Entity(model='quad', scale=(3, 3, 3), position=(0, -2), parent=woerterMenu)
pictLetter = Entity(model='quad', scale=(3, 3, 3), position=(0, -2), parent=woerterMenu)
liveLetter = Entity(model='quad', scale=(3, 3, 3), position=(0, -2), parent=woerterMenu)
woerterMenu.disable()

freeMenu = Entity()
freeMenu.position = (-1, 2)
Button(parent=freeMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Back', on_click=backToMain, position=(-3, 2))
Text(parent=freeMenu, text='Zeige folgenden Buchstaben: B', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=freeMenu, text='Erkannte Buchstaben:  B  C  D  F ...', color=color.black, scale=(15, 15), position=(-2, 0))
freeMenu.disable()

buchstabenMenu = Entity()
buchstabenMenu.position = (-1, 2)
Button(parent=buchstabenMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Back', on_click=backToMain, position=(-3, 2))
Text(parent=buchstabenMenu, text='Wort: ', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=buchstabenMenu, text='Erkannt: ', color=color.black, scale=(15, 15), position=(-2, 0))
buchstabenMenu.disable()

"""
livePreview = Entity()
livePreview.position = (-1, 2)
image_texture = load_texture(r'archive\alphabet_pictogram\A.jpg')
Text(parent=livePreview, text='Live Preview:', color=color.black, scale=(15, 15), position=(-2, -2))
"""


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

        it, frameBG = 0, None

        def update():
            global it, frameBG
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
                if it >= 25:
                    imgResized = skimage.transform.resize(extracted, (imageSize, imageSize, 3))
                    img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
                    evaluation = model(img)
                    argmax = np.argmax(evaluation)
                    bestLetter = Util.inverse(argmax)
                    bestEval = evaluation[0, argmax].numpy()
                    print("%s, %s" % (bestLetter, bestEval))
                    pictogram.texture = Texture(Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGBA), mode="RGBA"))
            it += 1

        app.run()
