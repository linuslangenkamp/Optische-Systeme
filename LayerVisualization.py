"""
Last Edit 18.01.2024
@author: Linus Langenkamp
"""

import numpy as np
import cv2
import skimage
from skimage.transform import resize
from tensorflow import keras

print("Packages imported...")
#%% import model

model = keras.models.load_model('models\CNN_NoBG_ext_TO.h5')
#%% define constants

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
#%%

layer_index = 1
output_layer = model.layers[layer_index].output
new_model = keras.models.Model(inputs=model.input, outputs=output_layer)
#%%

image_path = 'archive/asl_alphabet_train_noBG/asl_alphabet_train/F/F486.jpg'
loaded = cv2.imread(image_path)
imgResized = skimage.transform.resize(loaded, (imageSize, imageSize, 3))
img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
#%%

allImages = []
out = new_model.predict(img)
for idx in range(32):
    layer1Output = (255 * out[0, :, :, idx] / np.max(out[0, :, :, :])).astype('uint8')
    resized_image = cv2.resize(layer1Output, (200, 200), interpolation=cv2.INTER_NEAREST)
    allImages.append(resized_image)
#%% all images -> one big

bigImage = np.zeros((800, 1600), dtype="uint8")
for x in range(8):
    for y in range(4):
        bigImage[y * 200: ((y+1) * 200), x * 200: (x+1) * 200] = allImages[x + 8 * y]

#%%

while True:
    cv2.imshow('frame', bigImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
