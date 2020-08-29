from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tensorflow.keras as keras
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

global img
global filename
global text
global labels
global model

win = Tk()
win.title("Pneumonia Checker")
c = Canvas(win, width=400, height=400)
c.pack()
text = StringVar()
path = os.getcwd()
label = Label(win, textvariable=text)
label.pack()
height = 150
width = 150
path = r"C:\Users\domin\Desktop\BIAI\chest_data"
train_dir = os.path.join(path, 'train')
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=64,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(height, width),
                                                     class_mode='binary')

pathNames = train_data_gen.filenames

labels = []
for filename in pathNames:
    newLabel = filename.split('\\')[0]
    isNew = True
    for label in labels:
        if label == newLabel:
            isNew = False
            break
    if isNew:
        labels.append(newLabel)

height = 150
width = 150
pf = r"C:\Users\domin\Desktop\BIAI\chest_data\train\NORMAL\IM-0115-0001.jpeg"

model = tf.keras.models.load_model('model')

batch_holder = np.zeros((1, height, width, 3))


def loadPicture():
    global filename
    win.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpeg"), ("all files", "*.*")))
    global img
    img = Image.open(win.filename)
    img = ImageTk.PhotoImage(img.resize((150, 150), Image.ANTIALIAS))
    c.create_image(20, 20, anchor=NW, image=img)

    img = image.load_img(win.filename, target_size=(height, width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1, width, height, 3)
    result = model.predict(img, batch_size=1)
    index = 0
    for i in result:
        for j in i:
            if j == 1:
                break
            index += 1
    global text
    text.set(labels[index])


b = Button(win, text='Photo:', width=30, command=loadPicture)
b.pack()

win.mainloop()