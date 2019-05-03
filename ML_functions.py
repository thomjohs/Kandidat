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
from keras.models import model_from_json


def label_to_array(labels, label):
    if labels[19] == label:
        return True
    else:
        return False


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


def cm_to_percentage_total(cm):
    cm = cm.astype(dtype=np.float32)
    sum = 0
    for frame in cm:
        for value in frame:
            sum += value
    for i, frame in enumerate(cm):
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


def loadModel(jsonFile, weightFile):
    file = open("Model\\" + jsonFile, 'r')
    loadedModelFile = file.read()
    file.close()
    load_model = model_from_json(loadedModelFile)
    load_model.load_weights("Model\\" + weightFile)
    return load_model


def load_zero_mean_normalize_data_multiple_files(input_files, folder="ProcessedData\\"):
    data = load_data_multiple(input_files, folder)
    data_new, means, maxs = zero_mean_normalize_data(data)
    return data_new, means, maxs


def load_zero_mean_normalize_data_multiple_files_multiple_folders(input_files, folders=["ProcessedData\\"]):
    n = len(folders)
    data = load_data_multiple(input_files, folders[0])
    if n > 1:
        for i in range(1, n):
            data = np.vstack((data, load_data_multiple(input_files, folders[i])))
    data_new, means, maxs = zero_mean_normalize_data(data)
    return data_new, means, maxs


def load_zero_mean_normalize_data_folder(input_folder):
    data = load_folder(input_folder)
    data_new, means, maxs = zero_mean_normalize_data(data)
    return data_new, means, maxs


def load_zero_mean_normalize_data_multiple_folders(input_folders):
    n = len(input_folders)
    data = load_folder(input_folders[0])
    if n > 1:
        for i in range(1, n):
            data = np.vstack((data, load_folder(input_folders[i])))
    data_new, means, maxs = zero_mean_normalize_data(data)
    return data_new, means, maxs


def zero_mean_normalize_data(data):
    standardLen = 10
    data_array = np.asarray(data)
    numMatrix = data_array[:, :1]
    rangeMatrix = data_array[:, 1:standardLen * 1 + 1]
    dopplerMatrix = data_array[:, 1 + standardLen * 1:standardLen * 2 + 1]
    peakMatrix = data_array[:, 1 + standardLen * 2:standardLen * 3 + 1]
    xMatrix = data_array[:, 1 + standardLen * 3:standardLen * 4 + 1]
    yMatrix = data_array[:, 1 + standardLen * 4:standardLen * 5 + 1]
    labelMatrix = data_array[:, standardLen * 5 + 1:]

    ########Calculate with zero padding###########
    # Subtrackting all elemnts with the mean and normalizing
    # numMatrix_norm=zero_mean_normalized(numMatrix)
    # rangeMatrix_norm=zero_mean_normalized(rangeMatrix)
    # dopplerMatrix_norm=zero_mean(dopplerMatrix)
    # peakMatrix_norm=zero_mean_normalized(peakMatrix)
    # xMatrix_norm=zero_mean_normalized(xMatrix)
    # yMatrix_norm=zero_mean_normalized(yMatrix)

    ########Calculate without zero padding###########
    # To know what to subtract the mean from
    boolarr = np.zeros_like(rangeMatrix)
    for i in range(len(data_array)):
        n = int(numMatrix[i])
        boolarr[i] = np.hstack((np.ones(n), np.zeros(standardLen - n)))
    # Subtrackting non-padding variables with the mean and normalizing
    numMatrix_norm, numMean, numMax = zero_mean_normalized(numMatrix)
    elements = int(sum(numMatrix))
    rangeMatrix_norm, rangeMean, rangeMax = zero_mean_normalized_without_padding(rangeMatrix, elements, boolarr)
    dopplerMatrix_norm, dopplerMean, dopplerMax = zero_mean_normalized_without_padding(dopplerMatrix, elements, boolarr)
    peakMatrix_norm, peakMean, peakMax = zero_mean_normalized_without_padding(peakMatrix, elements, boolarr)
    xMatrix_norm, xMean, xMax = zero_mean_normalized_without_padding(xMatrix, elements, boolarr)
    yMatrix_norm, yMean, yMax = zero_mean_normalized_without_padding(yMatrix, elements, boolarr)

    data_new = np.hstack((numMatrix_norm, rangeMatrix_norm, dopplerMatrix_norm, peakMatrix_norm, xMatrix_norm,
                          yMatrix_norm, labelMatrix))
    means = {'numObj': numMean, 'range': rangeMean, 'doppler': dopplerMean, 'peak': peakMean, 'x': xMean, 'y': yMean}
    maxs = {'numObj': numMax, 'range': rangeMax, 'doppler': dopplerMax, 'peak': peakMax, 'x': xMax, 'y': yMax}
    return data_new, means, maxs


def zero_mean_normalized(arr):
    elements = len(arr) * len(arr[0])
    zero_mean = np.sum(arr) / elements
    arr_zero_mean = np.subtract(arr, zero_mean)
    max_of_arr_zero_mean = np.amax(np.absolute(arr_zero_mean))
    arr_normilized = np.divide(arr_zero_mean, max_of_arr_zero_mean)
    return arr_normilized, zero_mean, max_of_arr_zero_mean


def zero_mean_normalized_without_padding(arr, elements, boolarr):
    zero_mean = np.sum(arr) / elements
    arr_zero_mean = np.subtract(arr, np.multiply(boolarr, zero_mean))
    max_of_arr_zero_mean = np.amax(np.absolute(arr_zero_mean))
    arr_normilized = np.divide(arr_zero_mean, max_of_arr_zero_mean)
    return arr_normilized, zero_mean, max_of_arr_zero_mean


def zero_mean_normalize_data_frame(data, means, maxs):
    standardLen = 10
    data_array = np.asarray(data)
    numMatrix = data_array[:1]
    rangeMatrix = data_array[1:standardLen * 1 + 1]
    dopplerMatrix = data_array[1 + standardLen * 1:standardLen * 2 + 1]
    peakMatrix = data_array[1 + standardLen * 2:standardLen * 3 + 1]
    xMatrix = data_array[1 + standardLen * 3:standardLen * 4 + 1]
    yMatrix = data_array[1 + standardLen * 4:standardLen * 5 + 1]
    labelMatrix = data_array[standardLen * 5 + 1:]

    n = int(data[0])
    boolarr = np.hstack((np.ones(n), np.zeros(standardLen - n)))

    numMatrix = np.divide(np.subtract(numMatrix, means['numObj']), maxs['numObj'])
    rangeMatrix = np.divide(np.subtract(rangeMatrix, np.multiply(boolarr, means['range'])), maxs['range'])
    dopplerMatrix = np.divide(np.subtract(dopplerMatrix, np.multiply(boolarr, means['doppler'])), maxs['doppler'])
    peakMatrix = np.divide(np.subtract(peakMatrix, np.multiply(boolarr, means['peak'])), maxs['peak'])
    xMatrix = np.divide(np.subtract(xMatrix, np.multiply(boolarr, means['x'])), maxs['x'])
    yMatrix = np.divide(np.subtract(yMatrix, np.multiply(boolarr, means['y'])), maxs['y'])

    data_new = np.hstack((numMatrix, rangeMatrix, dopplerMatrix, peakMatrix, xMatrix, yMatrix, labelMatrix))
    return data_new.tolist()


def zero_mean_normalized_frame(arr, mean, max):
    arr_zero_mean = np.subtract(arr, mean)
    arr_normilized = np.divide(arr_zero_mean, max)
    return arr_normilized


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
    model.add(Conv1D(num_filters, kernel_size, batch_input_shape=(batch_size, time_steps, vector_size - 1),
                     activation='relu'))
    model.add(BatchNormalization(axis=1, scale=0))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(time_steps, vector_size - 1)))  # , batch_size=batch_size))
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
    model.add(Dropout(0.1))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(outputs, activation='softmax'))
    return model


def build_lstm_single_predict(time_steps, vector_size, outputs, batch_size, lstm_output, stateful):
    model = Sequential()
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(1, vector_size - 1),
                   batch_size=batch_size))
    model.add(Dropout(0.1))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(outputs, activation='softmax'))
    return model

# data_new, means, maxs = load_zero_mean_normalize_data_folder("ProcessedData")
# print(means)
# print(maxs)
