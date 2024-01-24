"""
Last Edit 21.01.2024
@author: Linus Langenkamp
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import skimage
from skimage.transform import resize

print("Packages imported...")
#%% get training data
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 2 * 901 * num_classes + 2 * 901
train_dir = 'archive/asl_alphabet_train_noBG/asl_alphabet_train/'


def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))

                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X, y


X_train, y_train = get_data(train_dir)
print("Images successfully imported...")
#%%

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)
#%%

print("The shape of one image is : ", X_train[0].shape)
#%%

plt.imshow(X_train[0])
plt.show()
#%%

X_data = X_train
y_data = y_train
print("Copies made...")
#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)
#%%
# One-Hot-Encoding the categorical data
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,29)
y_cat_test = to_categorical(y_test,29)

#%%
# Checking the dimensions of all the variables
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_cat_train.shape)
print(y_cat_test.shape)
#%%
# This is done to save CPU and RAM space while working on Kaggle Kernels. This will delete the specified data and save some space!
import gc
del X_data
del y_data
gc.collect()
#%%

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
print("Packages imported...")
#%%

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(29, activation='softmax'))

model.summary()
#%%

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
#%%

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#%%
# turn off warning in case you use cpu

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#%%

model.fit(X_train, y_cat_train,
          epochs=50,
          batch_size=64,
          verbose=2,
          validation_data=(X_test, y_cat_test),
         callbacks=[early_stop])
#%%

metrics = pd.DataFrame(model.history.history)
print("The model metrics are")
#%%

ax = metrics[['loss','val_loss']].plot(fontsize=14)
plt.yscale('log')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
plt.savefig(r'archive\images\netStats\lossLog.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()


#%%

metrics[['accuracy','val_accuracy']].plot()
plt.yscale('log')
plt.savefig(r'archive\images\netStats\accLog.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

#%%

model.evaluate(X_test,y_cat_test,verbose=0)
#%%

predict_x=model.predict(X_test)
predictions=np.argmax(predict_x,axis=1)
print("Predictions done...")
#%%

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
#%%

plt.figure(figsize=(12,12))
sns.heatmap(confusion_matrix(y_test,predictions),cmap='Blues', annot=True, cbar=True, square=True, fmt='g', linewidths=.5, annot_kws={"size": 10})
plt.savefig(r'archive\images\netStats\confusionMatrixNumbers.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

#%%

model.save('models\CNN_NoBG_ext_TO4.h5')
print("Model saved successfully...")
