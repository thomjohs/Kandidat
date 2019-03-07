import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D

vector_size = 0
max_feature_value = 0

#we'll need to group the frames
time_steps = 0

lstm_output = 0

num_filters = 0
kernel_size = 0


def load_data():
    x_t = 0
    y_t = 0
    x_v = 0
    y_v = 0
    return x_t, y_t, x_v, y_v


x_train, y_train, x_val, y_val = load_data()

model = Sequential()
model.add(Conv1D(num_filters,
                 kernel_size,
                 input_shape=(time_steps, vector_size),
                 activation='relu'))
model.add(LSTM(lstm_output, return_sequences=True))
model.add(LSTM(lstm_output))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
