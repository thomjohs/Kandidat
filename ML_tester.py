import supp
import tensorflow as tf
import ML_functions as ml
import array as arr
import datetime
from keras.preprocessing import sequence
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import csv

# GPU Tester
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

vector_size = 52
starttime = datetime.datetime.now()
input_file = "ArenSwipeNext1"
input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1",
               "ArenButton2", "ArenSlideUp2", "ArenSwipeNext2",
               "ArenButton3", "ArenSlideUp3", "ArenSwipeNext3",
               "AndreasButton1", "AndreasSlideUp1", "AndreasSwipeNext1",
               "AndreasButton2", "AndreasSlideUp2", "AndreasSwipeNext2",
               "AndreasButton3", "AndreasSlideUp3", "AndreasSwipeNext3",
               "AndreasButton4", "AndreasSlideUp4", "AndreasSwipeNext4",
               "AndreasButton5", "AndreasSlideUp5", "AndreasSwipeNext5",
               "GoodBackground1", "GoodBackground2"]

# Number of categories
outputs = 4

# training hyperparameters

epochs = 300
time_steps = 20
batch_size = 100

training_ratio = 0.7

# used in both models
lstm_output = 40
stateful = True

# only used in combined model
num_filters = 64
kernel_size = 5
repeats = 5

# for saving the model and weights
export = True
modelFile = "Arenmodel4.json"
weightFile = "Arenweights4.h5"

# saves plot
plot = False
plotFile = f'Plots\\ts{time_steps}bs{batch_size}lstmOut{lstm_output}st{stateful}.pdf'

# saves Result
resultFile = "resultsArencrrr4.csv"

data = supp.shuffle_gestures(ml.load_data_multiple(input_files))



gestFrames = 0
backFrames = 0
for frame in data:
    if len(frame) != 52:
        print(frame[51])
    if frame[len(frame) - 1] == 'goodBackground':
        backFrames += 1
    else:
        gestFrames += 1
        

x_train, x_test, y_train, y_test = ml.split_data(list(map(supp.label_to_int, data)), vector_size, outputs,
                                                 training_ratio)

x_train = x_train[:len(x_train) // 1000 * 1000 + time_steps]
x_test = x_test[:len(x_test) // 1000 * 1000 + time_steps]
y_train = y_train[:len(y_train) // 1000 * 1000 + time_steps]
y_test = y_test[:len(y_test) // 1000 * 1000 + time_steps]

print(f'{len(x_train)}, {len(x_test)}, {len(y_train)}, {len(y_test)}')


train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size)


seqtest=[]

for i in range(repeats):

    # model = ml.build_lstm(time_steps, vector_size, outputs, batch_size, lstm_output, stateful)
    # model = ml.build_clstm(time_steps, vector_size, outputs, num_filters, kernel_size, lstm_output)
    model = ml.build_crrr(time_steps, vector_size, outputs, num_filters, batch_size, kernel_size, lstm_output, stateful)


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit_generator(train_seq,
                                  callbacks=[LambdaCallback(
                                      on_epoch_begin=lambda epoch, logs: print('Repeats', i + 1, '/', repeats))],
                                  epochs=epochs,
                                  validation_data=test_seq)

    seqtest.append(model.evaluate_generator(test_seq))

    '''
	preds = list(model.predict_generator(test_seq))[:1000]
	predVer = list(y_test)[:1000]
	print(type(preds))
	print(preds[0])
	for i in range(len(preds)):
		pred = preds[i]
		preds[i] = tf.one_hot(tf.nn.top_k(pred).indices, tf.shape(pred)[0])
	print(type(preds))
	print(preds[0])
	print(type(y_test))
	print(predVer[0])
	print(classification_report(y_test[:len(preds)], preds))
	'''

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    #
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # if plot:
    #     plt.savefig(plotFile, bbox_inches='tight')
    # plt.show()

    print(f'Gestures: {gestFrames}')
    print(f'Backgrounds: {backFrames}')
    print(f'Percentage of gestures: {gestFrames / (gestFrames + backFrames)}')

    if export:
        json_model = model.to_json()
        with open("Model\\" + modelFile, 'w') as file:
            file.write(json_model)
        model.save_weights("Model\\" + weightFile)

with open(resultFile, 'w') as file:
    writer = csv.writer(file)
    for row in seqtest:
        writer.writerow(row)

ml.sum_print(starttime, repeats, seqtest)

#pyplot.plot(history['train'], color='blue')
#pyplot.plot(history['test'], color='orange')
#print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
