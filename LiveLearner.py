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
    freeMenu.disable()
    mainMenu.enable()


def beenden():
    app.destroy()
    app.userExit()


def sliderChange():
    global correctsInARow, holdingFrames
    correctsInARow = 0
    holdingFrames = max(1, int(holdDuration.value * 32))


def updateHoldBar(bar, value):
    value = clamp(value, 0, 1)
    bar.scale_x = 2 * value
    bar.x = -2 + value
    bar.color = lerp(color.red, color.green, value)


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
nnBorder = Entity(model='quad', color=color.white, scale=(5, 5), position=(1.5, 1), collider='box', parent=freeMenu)
nnBorder.roundness = 0.1
nnBackground = Entity(model='quad', color=color.rgb(55, 55, 55), scale=(4.9, 4.9), position=(1.5, 1, -1e-3), collider='box', parent=freeMenu)
liveLetterBaseF = Entity(model='quad', scale=(2, 2, 2), position=(6, 1), parent=freeMenu, color=color.white)
liveLetterF = Text(scale=(20, 20), origin=(0, 0), position=(0, 0, -1e-3), parent=liveLetterBaseF,  text="", color=color.black)
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
holdDurationB = Slider(parent=buchstabenMenu, position=(-6.3, -1.5), min=0, max=3, default=1, scale=(5, 5), on_value_changed=sliderChange)
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

        it, letterIdx, frameBG = 0, 0, None
        currentWord, currentLetter = None, None
        sliderChange()

        def update():
            global it, frameBG, currentWord, currentLetter, letterIdx, correctsInARow
            frame = handler.get_image()
            # reference background
            if it < 24:
                bglist.append(frame)
            elif it == 24:
                frameBG = np.mean(bglist, axis=0)
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
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    bgF.texture = Texture(
                        Image.fromarray(cv2.cvtColor(frameBG.astype('uint8'), cv2.COLOR_BGR2RGBA), mode="RGBA"))
                    liveLetterF.text = bestLetter
            it += 1

        app.run()
