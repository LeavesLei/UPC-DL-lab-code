"""
@author:Leaves
@file: 17flowers_cnn_lab_course.py
@time: 2019/03/01
"""
import os
import numpy as np
import keras

from numpy import random
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

dataset_name = '17flowers'
kernel_size = 11
acc_fig_name = "%s_%d_acc.jpg" % (dataset_name, kernel_size)
loss_fig_name = "%s_%d_loss.jpg" % (dataset_name, kernel_size)


def read_image(image_name):
    """
    covert image to numpy array
    :param image_name: image name
    :return: numpy array with the image
    """
    im = Image.open(image_name).convert('RGB')
    im = im.resize((224, 224))
    data = np.array(im)
    data = data / 255.
    return data


def load_data(data_path=os.path.abspath('.')):
    # read jpg name
    with open(data_path + "/jpg/files.txt", "r") as f:
        jpg_name = f.readlines()

    data_len = len(jpg_name)
    class_len = 80

    # image array list
    image_array = []

    # labels list
    label = []

    for i in range(data_len):
        jpg_name[i] = jpg_name[i].split('\n')[0]
        label.append((i // class_len))

        image_array.append(read_image(os.path.join('jpg', jpg_name[i])))
    return image_array, label


dataset, label = load_data()
input_shape = dataset[0].shape
dataset = np.array(dataset)
label = np.array(label)
# shuffle dataset
index = np.arange(dataset.shape[0])
random.shuffle(index)
dataset = dataset[index, :, :, :]
label = label[index]

num_classes = 17
# Adapt the labels to the one-hot vector syntax required by the softmax
label = np_utils.to_categorical(label, num_classes)

# Define the NN architecture : LeNet
model = Sequential()
model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

history = model.fit(dataset, label, validation_split=0.3, epochs=20, batch_size=8)

# store plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy (dataset: %s, kernel size: %d)" % (dataset_name, kernel_size))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(acc_fig_name)
plt.close()
# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss (dataset: %s, kernel size: %d)" % (dataset_name, kernel_size))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(loss_fig_name)
plt.close()
