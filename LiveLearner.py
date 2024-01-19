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

Text.default_font = 'fonts/Ubuntu-Regular.ttf'
Text.default_resolution = 200
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


def sliderChange():
    global correctsInARow, holdingFrames
    correctsInARow = 0
    holdingFrames = max(1, int(holdDuration.value * 32))


def update_health_bar(value):
    value = clamp(value, 0, 1)
    holdBar.scale_x = 2 * value
    holdBar.x = -2 + value
    holdBar.color = lerp(color.red, color.green, value)


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
Button(parent=woerterMenu, color=color.gray, scale=(3, 1), text='Zurück', on_click=backToMain, position=(-5, -3), radius=.2)
t1 = Text(parent=woerterMenu, text='', color=color.white, scale=(15, 15), position=(-5.53, 3.5))
t2 = Text(parent=woerterMenu, text='', color=color.white, scale=(15, 15), position=(-6, 2.5))
pictogram = Entity(model='quad', scale=(2, 2, 2), position=(-1, 2), parent=woerterMenu)
liveExtracted = Entity(model='quad', scale=(2, 2, 2), position=(-1, -1), parent=woerterMenu)
pictLetterBase = Entity(model='quad', scale=(2, 2, 2), position=(4, 2), parent=woerterMenu, color=color.white)
pictLetter = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=pictLetterBase,  text="", color=color.black)
liveLetterBase = Entity(model='quad', scale=(2, 2, 2), position=(4, -1), parent=woerterMenu, color=color.white)
liveLetter = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=liveLetterBase,  text="", color=color.black)
arrow = Entity(model='quad', scale=(3.74/2, 1.04/2), position=(1.5, -1.25), parent=woerterMenu, texture=load_texture("archive/images/pfeil.png"))
evalText = Text(scale=(20, 20), origin=(0, 0), position=(1.5, -0.75, -1e-3), parent=woerterMenu,  text="", color=color.white)
holdText = Text(scale=(15, 15), origin=(0, 0), position=(-5.05, -1), parent=woerterMenu, text="Haltezeit in sek.", color=color.white)
holdDuration = Slider(parent=woerterMenu, position=(-6.3, -1.5), min=0, max=3, default=1, scale=(5, 5), on_value_changed=sliderChange)
holdBar = Entity(model='quad', scale=(1, 0.1), color=color.red, position=(0, 0), parent=woerterMenu)
woerterMenu.disable()


freeMenu = Entity()
freeMenu.position = (-1, 2)
Button(parent=freeMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Zurück', on_click=backToMain, position=(-3, 2))
Text(parent=freeMenu, text='Zeige folgenden Buchstaben: B', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=freeMenu, text='Erkannte Buchstaben:  B  C  D  F ...', color=color.black, scale=(15, 15), position=(-2, 0))
freeMenu.disable()

buchstabenMenu = Entity()
buchstabenMenu.position = (-1, 2)
Button(parent=buchstabenMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Zurück', on_click=backToMain, position=(-3, 2))
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

        it, letterIdx, frameBG = 0, 0, None
        sliderChange()

        currentWord, currentLetter = None, None

        def update():
            global it, frameBG, currentWord, currentLetter, letterIdx, correctsInARow
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

                    bestLetter = Util.shortLetter(Util.inverse(argmax))
                    bestEval = evaluation[0, argmax].numpy()
                    print("%s, %s" % (bestLetter, bestEval))
                    if bestEval < 0.8:
                        bestLetter = "?"
                    elif bestEval < 0.99:
                        bestLetter += "?"

                    if woerterMenu.enabled:
                        if currentWord is None or letterIdx == len(currentWord):
                            currentWord = Util.generateWord()
                            letterIdx = 0
                            t1.text = 'Wort: ' + currentWord
                            t2.text = 'Erkannt: '
                        currentLetter = currentWord[letterIdx]
                        if bestLetter == currentLetter:
                            correctsInARow += 1
                        else:
                            correctsInARow = 0
                        update_health_bar(correctsInARow / holdingFrames)
                        if correctsInARow == holdingFrames:
                            t2.text += bestLetter
                            letterIdx += 1
                            correctsInARow = 0
                        liveExtracted.texture = Texture(Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGBA), mode="RGBA"))
                        pictogram.texture = load_texture(r"archive\alphabet_pictogram\%s.jpg" % currentLetter)
                        liveLetter.text = bestLetter
                        pictLetter.text = Util.shortLetter(currentLetter)
                        evalText.text = f"{bestEval:.03}"
            it += 1

        app.run()
