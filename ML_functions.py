import tensorflow as tf
import datetime
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Embedding
from keras import utils
from keras.layers import LSTM, Conv1D
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import supp
import os


def cm_to_percentage(cm):
    cm = cm.astype(dtype=np.float32)
    for i, frame in enumerate(cm):
        sum = 0
        for value in frame:
            sum += value
        for j, value in enumerate(frame):
            perc = float(value / sum) * 100
            cm[i][j] = perc
        cm[i] = np.around(frame, decimals=1)
    return cm


def sum_print(start_time, repeats, seqtest):
    print('')
    print('Validation Results:')
    for j in range(repeats):
        [score, acc] = seqtest.pop(0)
        print('Test score:', round(score, 3), 'Test acc:', round(acc, 3))

    print('')
    print('Time points:')
    endtime = datetime.datetime.now()
    dur = endtime - start_time
    print('Starttime:', start_time)
    print('Endtime:', endtime)
    print('Duration:', dur)


def load_folder(input_folder):
    for root, dirs, files in os.walk(input_folder):
        if '.csv' in files:
            files.remove('.csv')
        print(files)
        return load_data_multiple(files, input_folder + "\\")


def load_data_multiple(input_files, folder="ProcessedData\\"):
    frameList = []
    for file in input_files:
        frameList.extend(load_data(file, folder))
    return frameList


def load_data(input_file, folder="ProcessedData\\"):
    frameList = []
    with open(folder + input_file) as inp:
        reader = csv.reader(inp, delimiter=',')
        for row in reader:
            frame = row
            if len(frame) != 0:
                frameList.append(frame)
    data = np.empty((len(frameList), len(frameList[0])), dtype=np.float32)
    i = 0
    for frame in frameList:
        frame_int = supp.label_to_int(frame)
        data[i] = np.array(frame_int)
        i += 1
    return data

def count_gestures(data):
    gest_count = [0, 0, 0, 0, 0, 0, 0]
    for gesture in data:
        for i in range(len(gesture)):
            if gesture[i] == 1.0:
                gest_count[i] += 1
    return gest_count


def split_data(data, vector_size, outputs, training_ratio):
    training_n = int(len(data) * training_ratio)
    split = np.split(data, [vector_size - 1, vector_size], axis=1)
    x = split[0]
    y = split[1]
    y = utils.to_categorical(y, outputs, dtype=np.float32)

    x_split = np.split(x, [training_n, len(data)])
    y_split = np.split(y, [training_n, len(data)])

    return x_split[0], x_split[1], y_split[0], y_split[1]


def build_clstm(time_steps, vector_size, outputs, num_filters, kernel_size, lstm_output):
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, input_shape=(time_steps, vector_size - 1), activation='relu'))
    model.add(LSTM(lstm_output, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(lstm_output))
    model.add(Dense(outputs, activation='softmax'))
    return model


def build_crrr(time_steps, vector_size, outputs, num_filters, batch_size, kernel_size, lstm_output, stateful):
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, batch_input_shape=(batch_size, time_steps, vector_size - 1), activation='relu'))
    model.add(BatchNormalization(axis=1, scale=0))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(time_steps, vector_size - 1)))#, batch_size=batch_size))
    model.add(Dropout(0.1))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(outputs, activation='softmax'))
    return model


def build_lstm(time_steps, vector_size, outputs, batch_size, lstm_output, stateful):
    model = Sequential()
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(time_steps, vector_size - 1),
                   batch_size=batch_size))
    # model.add(Dropout(0.1))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(outputs, activation='softmax'))
    return model
