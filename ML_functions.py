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


def load_data_multiple(input_files):
    frameList = []
    for file in input_files:
        frameList.extend(load_data(file))
    return frameList


def load_data(input_file):
    frameList = []
    with open("ProcessedData\\" + input_file + ".csv") as inp:
        reader = csv.reader(inp, delimiter=',')
        for row in reader:
            frame = row
            if len(frame) != 0:
                frameList.append(frame)
    return frameList


def split_data(frameList, vector_size, outputs, training_ratio):
    x = np.empty((len(frameList), vector_size - 1), dtype=np.float32)
    y = np.empty((len(frameList), 1), dtype=np.float32)
    i = 0
    for frame in frameList:
        x[i] = np.array(frame[:vector_size - 1])
        y[i] = frame[vector_size - 1]
        i += 1
    y = utils.to_categorical(y, outputs, dtype=np.float32)
    x_train = x[:int(len(frameList) * training_ratio)]
    x_test = x[int(len(frameList) * training_ratio):len(frameList)]
    y_train = y[:int(len(frameList) * training_ratio)]
    y_test = y[int(len(frameList) * training_ratio):len(frameList)]
    return x_train, x_test, y_train, y_test


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
