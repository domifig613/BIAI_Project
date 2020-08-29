import numpy as np
import pandas as pd
import cv2
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import os

import os
for path, _, files in os.walk('C:/Users/domin/Desktop/BIAI'):
    for file in files:
        os.path.join(path, file)

tags = ['PNEUMONIA', 'NORMAL']
image_size = 150

def trainingData(data_dir):
    data = []
    for label in tags:
        path = os.path.join(data_dir, label)
        class_num = tags.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (image_size, image_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

test = trainingData('C:/Users/domin/Desktop/BIAI/chest_data/test')
val = trainingData('C:/Users/domin/Desktop/BIAI/chest_data/val')
train = trainingData('C:/Users/domin/Desktop/BIAI/chest_data/train')

l = []
for i in train:
    if(i[1] != 0):
        l.append("Normal")
    else:
        l.append("Pneumonia")
sns.set_style('darkgrid')
sns.countplot(l)

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(tags[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(tags[train[-1][1]])

x_test = []
y_test = []

x_train = []
y_train = []

x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, image_size, image_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, image_size, image_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, image_size, image_size, 1)
y_test = np.array(y_test)

dg = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20,
    zoom_range=0.02,
    width_shift_range=0.03,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False)

dg.fit(x_train)

mod = Sequential()
mod.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
mod.add(BatchNormalization())
mod.add(MaxPool2D((2, 2), strides=2, padding='same'))
mod.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
mod.add(Dropout(0.1))
mod.add(BatchNormalization())
mod.add(MaxPool2D((2, 2), strides=2, padding='same'))
mod.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
mod.add(BatchNormalization())
mod.add(MaxPool2D((2, 2), strides=2, padding='same'))
mod.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
mod.add(Dropout(0.2))
mod.add(BatchNormalization())
mod.add(MaxPool2D((2, 2), strides=2, padding='same'))
mod.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
mod.add(Dropout(0.2))
mod.add(BatchNormalization())
mod.add(MaxPool2D((2, 2), strides=2, padding='same'))
mod.add(Flatten())
mod.add(Dense(units=128, activation='relu'))
mod.add(Dropout(0.2))
mod.add(Dense(units=1, activation='sigmoid'))
mod.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
mod.summary()

lr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

his = mod.fit(dg.flow(x_train, y_train, batch_size=32), epochs=12, validation_data=dg.flow(x_val, y_val), callbacks=[lr])

print("Loss is - ", mod.evaluate(x_test, y_test)[0])
print("Accuracy is - ", mod.evaluate(x_test, y_test)[1] * 100, "%")

epochs = [i for i in range(12)]
fi, a = plt.subplots(1, 2)
train_acc = his.history['accuracy']
train_loss = his.history['loss']
val_acc = his.history['val_accuracy']
val_loss = his.history['val_loss']
fi.set_size_inches(20, 10)

a[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
a[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
a[0].set_title('Training & Validation Accuracy')
a[0].legend()
a[0].set_xlabel("Epochs")
a[0].set_ylabel("Accuracy")

a[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
a[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
a[1].set_title('Testing Accuracy & Loss')
a[1].legend()
a[1].set_xlabel("Epochs")
a[1].set_ylabel("Training & Validation Loss")
plt.show()

pre = mod.predict_classes(x_test)
pre = pre.reshape(1, -1)[0]
pre[:15]

print(classification_report(y_test, pre, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))

cm = confusion_matrix(y_test, pre)
cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])

plt.figure(figsize=(10, 10))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=tags, yticklabels=tags)

correct = np.nonzero(pre == y_test)[0]
incorrect = np.nonzero(pre != y_test)[0]

i = 0
for c in correct[:6]:
    plt.subplot(3, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150, 150), cmap="gray", interpolation='none')
    plt.title("Predicted {},Actual {}".format(pre[c], y_test[c]))
    plt.tight_layout()
    i += 1

i = 0
for ic in incorrect[:6]:
    plt.subplot(3, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[ic].reshape(150, 150), cmap="gray", interpolation='none')
    plt.title("Predicted {},Class {}".format(pre[ic], y_test[ic]))
    plt.tight_layout()
    i += 1
