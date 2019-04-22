from keras import utils
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix

import ManipuleraData as manip
import msvcrt
from tkinter import *
from pynput.keyboard import KeyCode, Controller



# ML variables set to the same as current model
batch_size = 500



# Model loading data
modelFile = "model.json"
weightFile = "weights.h5"

vector_size = 52
outputs = 7


epochs = 20
time_steps = 10
training_ratio = 0.7



def loadModel(jsonFile, weightFile):
    file = open("Model\\" + jsonFile, 'r')
    loadedModelFile = file.read()
    file.close()
    load_model = model_from_json(loadedModelFile)
    load_model.load_weights("Model\\" + weightFile)
    return load_model



model = loadModel(modelFile, weightFile)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

data = supp.shuffle_gestures(ml.load_folder("ProcessedData"))
x_train, x_test, y_train, y_test = ml.split_data(data, vector_size, outputs,
                                                 training_ratio)

x_train = x_train[:len(x_train) // 1000 * 1000 + time_steps]
x_test = x_test[:len(x_test) // 1000 * 1000 + time_steps]
y_train = y_train[:len(y_train) // 1000 * 1000 + time_steps]
y_test = y_test[:len(y_test) // 1000 * 1000 + time_steps]

train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size, shuffle=0)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size, shuffle=0)


predictions = model.predict_generator(test_seq)
print(predictions[0])

predictions_argmax = np.argmax(predictions, axis=1)
# predictions = utils.to_categorical(predictions, outputs, dtype=np.float32)

y_argmax = np.argmax(y_train[:len(y_test) // 1000 * 1000], axis=1)

cm = confusion_matrix(y_argmax, predictions_argmax)
print(cm)
cm = cm.astype(dtype=np.float32)

print(ml.cm_to_percentage(cm))
print(ml.cm_to_percentage_total(cm))