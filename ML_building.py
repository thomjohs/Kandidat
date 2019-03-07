import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D
import csv

vector_size = 0
time_steps = 0
batch_size = 0

input_file = "frameData.csv"

def load_data(input_file):
    with open(input_file) as csv_file:
        reader = csv.DictReader(csv_file)
        print(list(reader)[0])


    return x_t, y_t, x_v, y_v


x_train, y_train, x_test, y_test = load_data()


def build_clstm(num_filters, kernel_size, lstm_output):
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, input_shape=(time_steps, vector_size), activation='relu'))
    model.add(LSTM(lstm_output, return_sequences=True))
    model.add(LSTM(lstm_output))
    model.add(Dense(7, activation='softmax'))
    return model


def build_lstm(lstm_output, stateful=False):
    global batch_size
    model = Sequential()
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful,
                   input_shape=(time_steps, vector_size),
                   batch_size=batch_size))
    model.add(LSTM(lstm_output,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(lstm_output,
                   stateful=stateful))
    model.add(Dense(7, activation='softmax'))
    return model


model = build_lstm(64, False)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)