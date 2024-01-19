from ursina import *
import Util
import numpy as np
from PIL import Image
from matplotlib import cm


app = Ursina(title="sign language learning", borderless=False, icon="icon.ico")
#window.color = color.rgb(255, 255, 255)


def buchstaben():
    print("Buchstaben")
    mainMenu.disable()
    buchstabenMenu.enable()

def woerter():
    print("Wörter")
    mainMenu.disable()
    woerterMenu.enable()

def backToMain():
    woerterMenu.disable()
    buchstabenMenu.disable()
    mainMenu.enable()


mainMenu = Entity()
mainMenu.position = (-1, 2)
Button(parent=mainMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Buchstaben', on_click=buchstaben, position=(-2, 0.5))
Button(parent=mainMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Wörter', on_click=woerter, position=(2, 0.5))

woerterMenu = Entity()
woerterMenu.position = (-1, 2)
Button(parent=woerterMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Back', on_click=backToMain, position=(-3, 2))
Text(parent=woerterMenu, text='Zeige folgenden Buchstaben: B', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=woerterMenu, text='Erkannte Buchstaben:  B  C  D  F ...', color=color.black, scale=(15, 15), position=(-2, 0))
woerterMenu.disable()

buchstabenMenu = Entity()
buchstabenMenu.position = (-1, 2)
Button(parent=buchstabenMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Back', on_click=backToMain, position=(-3, 2))
Text(parent=buchstabenMenu, text='Zeige folgendes Wort: TEST', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=buchstabenMenu, text='Erkannte Buchstaben:  T  E  S  ...', color=color.black, scale=(15, 15), position=(-2, 0))
buchstabenMenu.disable()

livePreview = Entity()
livePreview.position = (-1, 2)
image_texture = load_texture(r'archive\alphabet_pictogram\A.jpg')
Text(parent=livePreview, text='Live Preview:', color=color.black, scale=(15, 15), position=(-2, -2))
pictogram = Entity(model='quad', texture=image_texture, scale=(6, 3, 3), position=(0, -2))
i = 0
test, test2 = None, None


bigImage = np.load('bigImage.npy')
bigImageExpanded = np.expand_dims(bigImage, axis=2)
bigImage = np.repeat(bigImageExpanded, 3, axis=2)
saturationValues = np.uint8(np.ones((800, 1600, 1)) * 255)
print(bigImage.shape, saturationValues.shape)


def update():
    global test, test2
    new_array = np.concatenate((bigImage, saturationValues), axis=2)
    im = Image.fromarray(new_array)
    # set pil image to texture
    pictogram.texture = Texture(im)

app.run()
