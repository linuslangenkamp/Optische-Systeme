#%%
"""
train_len = 87000
train_dir = 'archive/asl_alphabet_train/asl_alphabet_train/'


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
#%%

evaluation = model(at(X_train, 1))
print(inverse(np.argmax(evaluation)))
#%% Evaluate all standard test data
for i in range(ord("A"), ord("Z") + 1):
    imgRaw = cv2.imread('archive/asl_alphabet_test/asl_alphabet_test/%s_test.jpg' % chr(i))
    imgResized = skimage.transform.resize(imgRaw, (imageSize, imageSize, 3))
    img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
    evaluation = model(img)
    if inverse(np.argmax(evaluation)) ==  chr(i):
        print("True")
    else:
        print("False")
    print(inverse(np.argmax(evaluation)))
#%% import custom image
imgRaw = cv2.imread('archive/asl_self/L1.jpeg')
imgResized = skimage.transform.resize(imgRaw, (imageSize, imageSize, 3))
img = np.asarray(imgResized).reshape((-1, imageSize, imageSize, 3))
skimage.io.imshow(imgResized)
skimage.io.show()
evaluation = model(img)
allEvaluations(evaluation)
"""