import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras import utils
from keras.layers import LSTM, Conv1D
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import supp

vector_size = 52


input_file = "frameData.csv"

# Number of categories
outputs = 3

# training hyperparameters
epochs = 5
time_steps = 20
batch_size = 20
training_ratio = 0.8

# used in both models
lstm_output = 12
stateful = False

# only used in combined model
num_filters = 64
kernel_size = 5


def load_data(input_file):
    frameList = []
    with open(input_file) as inp:
        for row in inp:
            frame = supp.dString_to_farray(row)
            frameList.append(frame)
    return frameList

def label_to_int(frame):
    last = len(frame)-1
    if frame[last] == 'slideUp':
        frame[last] = 0
    elif frame[last] == 'button':
        frame[last] = 1
    else:
        frame[last] = 2
    return frame



def noise(num):
    num += num/2*random.randint(-1, 1)
    return num


def create_gesture1(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for j in range(vector_size):
            frame.append(noise(j))
        frame.append('slideUp')
        frameList.append(frame)
    return frameList


def create_gesture2(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(noise(11))
        frame.append('button')
        frameList.append(frame)
    return frameList


def create_gesture3(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(random.randint(0, 40))
        frame.append('swipe')
        frameList.append(frame)
    return frameList


def create_data():
    frameList = []
    for i in range(135):
        randGest = random.randint(1, 2)
        randAmount = 40 + random.randint(-10, 10)
        if randGest == 1:
            frameList.extend(create_gesture1(randAmount))
        elif randGest == 2:
            frameList.extend(create_gesture2(randAmount))
        else:
            frameList.extend(create_gesture3(randAmount))

    return frameList[:4000]



def split_data(frameList):
    x = np.empty((len(frameList), vector_size-1), dtype=np.float32)
    y = np.empty((len(frameList), 1), dtype=np.float32)
    i = 0
    for frame in frameList:
        x[i] = np.array(frame[:vector_size-1])
        y[i] = frame[vector_size]
        i += 1
    y = utils.to_categorical(y, outputs, dtype=np.float32)
    x_train = x[:int(len(frameList)*training_ratio)]
    x_test = x[int(len(frameList)*training_ratio):len(frameList)]
    y_train = y[:int(len(frameList)*training_ratio)]
    y_test = y[int(len(frameList)*training_ratio):len(frameList)]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_data(list(map(label_to_int, create_data())))

train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size)


def build_clstm():
    global num_filters
    global kernel_size
    global lstm_output
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, input_shape=(time_steps, vector_size-1), activation='relu'))
    model.add(LSTM(lstm_output, return_sequences=True))
    model.add(LSTM(lstm_output))
    model.add(Dense(outputs, activation='softmax'))
    return model


def build_lstm():
    global batch_size
    global lstm_output
    global stateful
    model = Sequential()
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(time_steps, vector_size - 1),
                   batch_size=batch_size))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(outputs, activation='softmax'))
    return model


model = build_lstm()
# model = build_clstm()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(train_seq, epochs=epochs,
                              validation_data=test_seq)

score, acc = model.evaluate_generator(test_seq)
print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predict_seq = test_seq = sequence.TimeseriesGenerator(x_test[100:200], y_test[100:200], length=time_steps, batch_size=batch_size)
predict = model.predict_generator(predict_seq, verbose=1)

