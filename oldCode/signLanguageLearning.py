from ursina import *

app = Ursina(title="sign language learning", borderless=False, icon="icon.ico")
window.exit_button.enabled = False
window.cog_button.enabled = False
window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.collider_counter.enabled = False
window.color = color.rgb(255, 255, 255)

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
Text(parent=woerterMenu, text='Zeige folgendes Wort: TEST', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=woerterMenu, text='Erkannte Buchstaben:  T  E  S  ...', color=color.black, scale=(15, 15), position=(-2, 0))
woerterMenu.disable()

buchstabenMenu = Entity()
buchstabenMenu.position = (-1, 2)
Button(parent=buchstabenMenu, model='quad', color=color.gray, scale=(3, 0.6), text='Back', on_click=backToMain, position=(-3, 2))
Text(parent=buchstabenMenu, text='Zeige folgenden Buchstaben: B', color=color.black, scale=(15, 15), position=(-2, 1))
Text(parent=buchstabenMenu, text='Erkannte Buchstaben:  B  C  D  F ...', color=color.black, scale=(15, 15), position=(-2, 0))
buchstabenMenu.disable()

livePreview = Entity()
livePreview.position = (-1, 2)
image_texture = load_texture('hand3.jpg')
Text(parent=livePreview, text='Live Preview:', color=color.black, scale=(15, 15), position=(-2, -2))
Entity(model='quad', texture=image_texture, scale=(3, 3, 3), position=(0, -2))


app.run()
