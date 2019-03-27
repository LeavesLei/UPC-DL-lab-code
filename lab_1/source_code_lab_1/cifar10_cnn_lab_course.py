"""
@author:Leaves
@file: cifar10_cnn_lab_course.py 
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
from keras.datasets import cifar10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

dataset_name = 'cifar10'
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

input_shape = x_train.shape[1:]

def train_and_test(dataset_name='cifar10', kernel_size=3):

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

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=8, shuffle=True)

    # store plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    acc_fig_name = "%s_%d_acc.jpg" % (dataset_name, kernel_size)
    loss_fig_name = "%s_%d_loss.jpg" % (dataset_name, kernel_size)
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

for kernel_size in [3, 5, 7, 9, 11]:
    print("Begin with kernel size: %d" % kernel_size)
    train_and_test(dataset_name=dataset_name, kernel_size=kernel_size)
    print('='*20)

