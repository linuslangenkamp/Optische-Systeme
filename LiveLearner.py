"""
Last Edit 21.01.2024
@author: Linus Langenkamp
"""

from ursina import *
import Util
import numpy as np
from PIL import Image
from tensorflow import keras
from vmbpy import *
import cv2
import skimage
from skimage.transform import resize

#%% import model

model = keras.models.load_model('models\CNN_NoBG_ext_TO2.h5')
modelC1 = keras.models.Model(inputs=model.input, outputs=model.layers[2].output)
modelC2 = keras.models.Model(inputs=model.input, outputs=model.layers[4].output)
modelC3 = keras.models.Model(inputs=model.input, outputs=model.layers[6].output)
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
    freeMenu.disable()
    mainMenu.enable()


def beenden():
    app.destroy()
    app.userExit()


def sliderChange():
    global correctsInARow, holdingFrames
    correctsInARow = 0
    holdingFrames = max(1, int(holdDuration.value * 32))


def sliderChangeB():
    global correctsInARow, holdingFrames
    correctsInARow = 0
    holdingFrames = max(1, int(holdDurationB.value * 32))


def sliderReset():
    global correctsInARow, holdingFrames
    correctsInARow = 0
    holdDurationB.value = 1
    holdDuration.value = 1
    holdingFrames = 32


def updateHoldBar(bar, value):
    value = clamp(value, 0, 1)
    bar.scale_x = 2 * value
    bar.x = -2 + value
    bar.color = lerp(color.red, color.green, value)


def interpolation():
    global interpBool
    interpBool = not interpBool
    interp.text = "Interpolation an" if interpBool else "Interpolation aus"


mainMenu = Entity()
mainMenu.position = (0, 0)
headline = Text(parent=mainMenu, scale=(25, 25), position=(0, 2), text="Deutsches Fingeralphabet - LiveLearner")
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Buchstaben', on_click=buchstaben, position=(-4, 0), radius=.2)
headline.setPos((-headline.width/2, 2, 0))
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Wörter', on_click=woerter, position=(0, 0), radius=.2)
Button(parent=mainMenu, color=color.gray, scale=(3.5, 1), text='Freier Modus', on_click=free, position=(4, 0), radius=.2)
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
freeMenu.position = (0, 0)
Button(parent=freeMenu, color=color.gray, scale=(3, 1), text='Zurück', on_click=backToMain, position=(-5, -3), radius=.2)
liveExtractedF = Entity(model='quad', scale=(2, 2, 2), position=(-3, 1), parent=freeMenu)
liveF = Entity(model='quad', scale=(2, 2, 2), position=(-6, 2.5), parent=freeMenu)
bgF = Entity(model='quad', scale=(2, 2, 2), position=(-6, -0.5), parent=freeMenu)
nnBorder = Entity(model='quad', color=color.white, scale=(5.1, 5.1), position=(1.5, 1), collider='box', parent=freeMenu)
nnBorder.roundness = 0.1
nnBackground = Entity(model='quad', color=color.rgb(55, 55, 55), scale=(5, 5), position=(1.5, 1, -1e-3), collider='box', parent=freeMenu)
Text(scale=(8, 8), position=(1.2, -1.8, 0), parent=freeMenu, text="Layer 2", color=color.white)
Text(scale=(8, 8), position=(1.2 - 5/3, -1.8, 0), parent=freeMenu, text="Layer 1", color=color.white)
Text(scale=(8, 8), position=(1.2 + 5/3, -1.8, 0), parent=freeMenu, text="Layer 3", color=color.white)
liveLetterBaseF = Entity(model='quad', scale=(2, 2, 2), position=(6, 1), parent=freeMenu, color=color.white)
liveLetterF = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=liveLetterBaseF,  text="", color=color.black)
interp = Button(text="Interpolation aus", on_click=interpolation, position=(1.5, -3, 0), scale=(5, 1), parent=freeMenu, color=color.gray)
Text(scale=(8, 8), position=(4.15, 1.35), parent=freeMenu, text="Output", color=color.white)
Entity(model='quad', scale=(3.74/6, 1.04/6), position=(4.5, 1), parent=freeMenu, texture=load_texture("archive/images/pfeil.png"))
Text(scale=(8, 8), position=(-1.75, 1.35), parent=freeMenu, text="Input", color=color.white)
Entity(model='quad', scale=(3.74/6, 1.04/6), position=(-1.5, 1), parent=freeMenu, texture=load_texture("archive/images/pfeil.png"))
Text(scale=(8, 8), position=(0.1, 3.9), parent=freeMenu, text="Convolutional Neural Network", color=color.white)
Entity(model='quad', scale=(3.4/6, 3.4/6), position=(-4.5, 2), parent=freeMenu, texture=load_texture("archive/images/pfeil_45.png"))
Entity(model='quad', scale=(3.4/6, 3.4/6), position=(-4.5, 0), parent=freeMenu, texture=load_texture("archive/images/pfeil45.png"))
Text(scale=(8, 8), position=(-6.95, 0.8), parent=freeMenu, text="Referenzhintergrund", color=color.white)
Text(scale=(8, 8), position=(-6.37, 3.8), parent=freeMenu, text="Livebild", color=color.white)
Text(scale=(8, 8), position=(-3.79, 2.3), parent=freeMenu, text="isoliertes Livebild", color=color.white)
Text(scale=(8, 8), position=(5.4, 2.3), parent=freeMenu, text="Klassifikation", color=color.white)
freeMenu.disable()

buchstabenMenu = Entity()
buchstabenMenu.position = (0, 0)
Button(parent=buchstabenMenu, color=color.gray, scale=(3, 1), text='Zurück', on_click=backToMain, position=(-5, -3), radius=.2)
pictogramB = Entity(model='quad', scale=(2, 2, 2), position=(-1, 2), parent=buchstabenMenu)
liveExtractedB = Entity(model='quad', scale=(2, 2, 2), position=(-1, -1), parent=buchstabenMenu)
pictLetterBaseB = Entity(model='quad', scale=(2, 2, 2), position=(4, 2), parent=buchstabenMenu, color=color.white)
pictLetterB = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=pictLetterBaseB,  text="", color=color.black)
liveLetterBaseB = Entity(model='quad', scale=(2, 2, 2), position=(4, -1), parent=buchstabenMenu, color=color.white)
liveLetterB = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=liveLetterBaseB,  text="", color=color.black)
arrowB = Entity(model='quad', scale=(3.74/2, 1.04/2), position=(1.5, -1.25), parent=buchstabenMenu, texture=load_texture("archive/images/pfeil.png"))
evalTextB = Text(scale=(20, 20), origin=(0, 0), position=(1.5, -0.75, -1e-3), parent=buchstabenMenu,  text="", color=color.white)
holdTextB = Text(scale=(15, 15), origin=(0, 0), position=(-5.05, -1), parent=buchstabenMenu, text="Haltezeit in sek.", color=color.white)
holdDurationB = Slider(parent=buchstabenMenu, position=(-6.3, -1.5), min=0, max=3, default=1, scale=(5, 5), on_value_changed=sliderChangeB)
holdBarB = Entity(model='quad', scale=(1, 0.1), color=color.red, position=(0, 0), parent=buchstabenMenu)
buchstabenMenu.disable()


with VmbSystem.get_instance() as vmb:
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
        interpBool = False
        it, letterIdx, frameBG = 0, 0, None
        currentWord, currentLetter = None, None
        c1 = None
        sliderChange()

        def update():
            global it, frameBG, currentWord, currentLetter, letterIdx, correctsInARow, c1, holdingFrames
            frame = handler.get_image()
            # reference background
            if it < 24:
                bglist.append(frame)
            elif it == 24:
                frameBG = np.mean(bglist, axis=0)
                bgF.texture = Texture(
                    Image.fromarray(cv2.cvtColor(frameBG.astype('uint8'), cv2.COLOR_BGR2RGBA), mode="RGBA"))
            else:
                # extraction, resizing, evaluation, best match
                extracted = Util.extractBG(frame, frameBG, kernelSmall, kernelBig)
                extracted = cv2.resize(extracted, (200, 200))
                imgResized = skimage.transform.resize(extracted, (imageSize, imageSize, 3))
                img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
                evaluation = model(img)
                argmax = np.argmax(evaluation)
                bestLetter = Util.shortLetter(Util.inverse(argmax))
                bestEval = evaluation[0, argmax].numpy()

                # ? if uncertain
                if bestEval < 0.8:
                    bestLetter = "?"
                elif bestEval < 0.99:
                    bestLetter += "?"

                if not (woerterMenu.enabled or buchstabenMenu.enabled):
                    sliderReset()

                # ui cases
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
                    updateHoldBar(holdBar, correctsInARow / holdingFrames)
                    if correctsInARow == holdingFrames:
                        t2.text += bestLetter
                        letterIdx += 1
                        correctsInARow = 0
                    liveExtracted.texture = Texture(
                        Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    pictogram.texture = load_texture(r"archive\alphabet_pictogram\%s.jpg" % currentLetter)
                    liveLetter.text = bestLetter
                    pictLetter.text = Util.shortLetter(currentLetter)
                    evalText.text = f"{bestEval:.03}"

                elif buchstabenMenu.enabled:
                    if currentLetter is None:
                        currentLetter = Util.generateLetter()
                    if bestLetter == currentLetter:
                        correctsInARow += 1
                    else:
                        correctsInARow = 0
                    updateHoldBar(holdBarB, correctsInARow / holdingFrames)
                    if correctsInARow == holdingFrames:
                        correctsInARow = 0
                        currentLetter = Util.generateLetter()
                    liveExtractedB.texture = Texture(
                        Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    pictogramB.texture = load_texture(r"archive\alphabet_pictogram\%s.jpg" % currentLetter)
                    liveLetterB.text = bestLetter
                    pictLetterB.text = Util.shortLetter(currentLetter)
                    evalTextB.text = f"{bestEval:.03}"
                elif freeMenu.enabled:
                    liveExtractedF.texture = Texture(
                        Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    liveF.texture = Texture(
                        Image.fromarray(cv2.cvtColor(cv2.resize(frame, (200, 200)), cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    liveLetterF.text = bestLetter

                    if interpBool:
                        iMode = cv2.INTER_LINEAR
                    else:
                        iMode = cv2.INTER_NEAREST

                    cImages = []
                    c1 = modelC1(img).numpy()
                    for idx in range(12):
                        layer1Output = (255 * c1[0, :, :, idx] / np.max(c1[0, :, :, :]))
                        cImages.append(cv2.resize(layer1Output, (100, 100), interpolation=iMode))

                    c2 = modelC2(img).numpy()
                    for idx in range(12):
                        layer2Output = (255 * c2[0, :, :, idx] / np.max(c2[0, :, :, :]))
                        cImages.append(cv2.resize(layer2Output, (100, 100), interpolation=iMode))

                    c3 = modelC3(img).numpy()
                    for idx in range(12):
                        layer3Output = (255 * c3[0, :, :, idx] / np.max(c3[0, :, :, :]))
                        cImages.append(cv2.resize(layer3Output, (100, 100), interpolation=iMode))

                    bigImage = np.zeros((600, 600), dtype="uint8")
                    for x in range(6):
                        for y in range(6):
                            bigImage[x * 100: (x + 1) * 100, y * 100: ((y + 1) * 100)] = cImages[x + 6 * y]
                    nnBackground.texture = Texture(
                        Image.fromarray(cv2.cvtColor(bigImage, cv2.COLOR_BGR2RGBA), mode="RGBA"))

            it += 1

        app.run()
