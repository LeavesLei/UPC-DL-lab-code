import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
import numpy as np
print( 'Using Keras version', keras.__version__)

def load_data(path='./mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
#Load the MNIST dataset, already provided by Keras
(x_train, y_train), (x_test, y_test) = load_data()

#Check sizes of dataset
print( 'Number of train examples', x_train.shape[0])
print( 'Size of train examples', x_train.shape[1:])


#Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

nlayers= 1
RNN = LSTM
neurons_list = [4, 8, 16]
drop_list= [0, 0.5]
impl=1

for neurons in neurons_list:
    for drop in drop_list:
        #Define the NN architecture
        model = Sequential()
        if nlayers == 1:
            model.add(RNN(neurons, input_shape=(x_train.shape[1], x_train.shape[2]), implementation=impl))
        else:
            model.add(RNN(neurons, input_shape=(x_train.shape[1], x_train.shape[2]), implementation=impl,
                          return_sequences=True))
            for i in range(1, nlayers - 1):
                model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

        model.add(Dense(10, activation='softmax')) 
        model.summary()

        #Model visualization
        #We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
        #from keras.util import plot_model
        #plot_model(nn, to_file='nn.png', show_shapes=true)

        #Compile the NN
        model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

        #Start training
        history = model.fit(x_train,y_train, validation_data=(x_test, y_test), batch_size=128, epochs=200)

        ##Store Plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #Accuracy plot
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy-neurons=%s-dropout=%s" % (neurons, drop))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper left')
        plt.savefig("mnist_cnn_accuracy-neurons=%s-dropout=%s.png" % (neurons, drop))
        plt.close()
        #Loss plot
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("model loss-neurons=%s-dropout=%s" % (neurons, drop))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper left')
        plt.savefig("mnist_cnn_loss-neurons=%s-dropout=%s.png" % (neurons, drop))
        plt.close()