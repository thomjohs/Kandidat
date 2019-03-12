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

vector_size = 52
time_steps = 10
batch_size = 10

input_file = "frameData.csv"
outputs = 3

def load_data(input_file):
   pass


def noise(num):
    num += num/5*random.randint(-1, 1)
    return num


def create_gesture1(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for i in range(vector_size):
            frame.append(noise(i))
        frame.append(0)
        frameList.append(frame)
    return frameList

def create_gesture2(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(noise(11))
        frame.append(1)
        frameList.append(frame)
    return frameList

def create_gesture3(amount):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(random.randint(0, 40))
        frame.append(2)
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
    x = np.empty((4000, 51), dtype=np.float32)
    y = np.empty((4000, 1), dtype=np.float32)
    i = 0
    for frame in frameList:
        x[i] = np.array(frame[:51])
        y[i] = frame[52]
        i += 1
    y = utils.to_categorical(y, outputs, dtype=np.float32)
    x_train = x[:3500]
    x_test = x[3500:4000]
    y_train = y[:3500]
    y_test = y[3500:4000]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_data(create_data())

train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size)

#print(x_train[0])
#print(test_seq[0])

def build_clstm(num_filters, kernel_size, lstm_output):
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, input_shape=(time_steps, vector_size-1), activation='relu'))
    model.add(LSTM(lstm_output, return_sequences=True))
    model.add(LSTM(lstm_output))
    model.add(Dense(outputs, activation='softmax'))
    return model


def build_lstm(lstm_output, stateful=False):
    global batch_size
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


model = build_lstm(64, False)
#model = build_clstm(64, 5, 64)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(train_seq, epochs=5,
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