import supp
import ML_functions as ml
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np


vector_size = 52

input_file = "ArenSwipeNext1"
input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1",
               "ArenButton2", "ArenSlideUp2", "ArenSwipeNext2",
               "AndreasButton1", "AndreasSlideUp1", "AndreasSwipeNext1",
               "AndreasButton2", "AndreasSlideUp2", "AndreasSwipeNext2",
               "GoodBackground1", "GoodBackground2"]

# Number of categories
outputs = 4

# training hyperparameters
epochs = 10
time_steps = 5
batch_size = 40
training_ratio = 0.7

# used in both models
lstm_output = 20
stateful = False

# only used in combined model
num_filters = 64
kernel_size = 5

# for saving the model and weights
export = False
modelFile = "model.json"
weightFile = "weights.h5"

# saves plot
plot = False
plotFile = f'Plots\\ts{time_steps}bs{batch_size}lstmOut{lstm_output}st{stateful}.pdf'

data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
data = data[:len(data)//1000 * 1000]


gestFrames = 0
backFrames = 0
for frame in data:
    if len(frame) != 52:
        print(frame[51])
    if frame[len(frame) -1] == 'goodBackground':
        backFrames += 1
    else:
        gestFrames += 1




x_train, x_test, y_train, y_test = ml.split_data(list(map(supp.label_to_int, data)), vector_size, outputs, training_ratio)

train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size)


# model = ml.build_lstm(time_steps, vector_size, outputs, batch_size, lstm_output, stateful)
model = ml.build_clstm(time_steps, vector_size, outputs, num_filters, kernel_size, lstm_output)

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
if plot:
    plt.savefig(plotFile, bbox_inches='tight')
plt.show()

print(f'Gestures: {gestFrames}')
print(f'Backgrounds: {backFrames}')
print(f'Percentage of gestures: {gestFrames/(gestFrames+backFrames)}')

if export:
    json_model = model.to_json()
    with open("Model\\" + modelFile, 'w') as file:
        file.write(json_model)
    model.save_weights("Model\\" + weightFile)




