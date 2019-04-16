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
import ManipuleraData as mani
from sklearn.metrics import confusion_matrix

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
               "GoodBackground1", "GoodBackground2", "AlexButton1", "AlexFlop1",
               "AlexSlideDown1", "AlexSlideUp1", "AlexSwipeNext1", "AlexSwipePrev1","JuliaButton1", "JuliaFlop1","JuliaSlideDown1", 
               "JuliaSlideUp1", "JuliaSwipeNext1", "JuliaSwipePrev1",
               "LinusButton1", "LinusFlop1", "LinusSlideDown1",
               "LinusSlideUp1", "LinusSwipeNext1", "LinusSwipePrev1",
               "MartinButton1", "MartinFlop1", "MartinSlideDown1",
               "MartinSlideUp1", "MartinSwipeNext1", "MartinSwipePrev1",
               "MatildaButton1", "MatildaFlop1", "MatildaSlideDown1",
               "MatildaSlideUp1", "MatildaSwipeNext1", "MatildaSwipePrev1"]
input_folder = "ProcessedData"
art_folder = "TranslatedData"

# Number of categories
outputs = 7

# training hyperparameters

epochs = 200
time_steps = 20
batch_size = 500

training_ratio = 0.7

# used in both models
lstm_output = 20
stateful = True

# only used in combined model
num_filters = 64
kernel_size = 5
repeats = 1

# for saving the model and weights
export = True
modelFile = "model.json"
weightFile = "weights.h5"

# saves plot
plot = False
plotFile = f'Plots\\ts{time_steps}bs{batch_size}lstmOut{lstm_output}st{stateful}.pdf'

# saves Result
resultFile = "results.csv"

data = supp.shuffle_gestures(ml.load_folder(input_folder))

# art_data = ml.load_folder(art_folder)
# art_background = ml.load_data("GoodBackground1.csv")

# art_data = supp.shuffle_gestures(np.concatenate
#                                ([art_data, art_background], axis=0))






x_train, x_test, y_train, y_test = ml.split_data(data, vector_size, outputs,
                                                 training_ratio)

# art_x, _, art_y, _ = ml.split_data(art_data, vector_size, outputs, 1)

# x_train = np.concatenate([x_train, art_x], axis=0)
# y_train = np.concatenate([y_train, art_y], axis=0)

x_train = x_train[:len(x_train) // 1000 * 1000 + time_steps]
x_test = x_test[:len(x_test) // 1000 * 1000 + time_steps]
y_train = y_train[:len(y_train) // 1000 * 1000 + time_steps]
y_test = y_test[:len(y_test) // 1000 * 1000 + time_steps]


print(ml.count_gestures(y_train))
print(ml.count_gestures(y_test))

print(f'{len(x_train)}, {len(x_test)}, {len(y_train)}, {len(y_test)}')


train_seq = sequence.TimeseriesGenerator(x_train, y_train, length=time_steps, batch_size=batch_size)
test_seq = sequence.TimeseriesGenerator(x_test, y_test, length=time_steps, batch_size=batch_size)


seqtest = []

pltloss = plt
pltacc = plt


for i in range(repeats):

    model = ml.build_lstm(time_steps, vector_size, outputs, batch_size, lstm_output, stateful)
    # model = ml.build_clstm(time_steps, vector_size, outputs, num_filters, kernel_size, lstm_output)
    # model = ml.build_crrr(time_steps, vector_size, outputs, num_filters, batch_size, kernel_size, lstm_output, stateful)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit_generator(train_seq,
                                  callbacks=[LambdaCallback(
                                      on_epoch_begin=lambda epoch, logs: print('Repeats', i + 1, '/', repeats))],
                                  epochs=epochs,
                                  validation_data=test_seq)

    seqtest.append(model.evaluate_generator(test_seq))

    predictions = model.predict_generator(train_seq)
    predictions = np.argmax(predictions, axis=1)
    cm = confusion_matrix(np.argmax(y_train[:len(y_train) // 1000 * 1000], axis=1), predictions)
    print(cm)

    cm = ml.cm_to_percentage(cm)
    print(cm)
    with open("ConfusionMatrix_dropout.csv", 'w', newline='') as cm_file:
        writer = csv.writer(cm_file)
        for row in cm:
            writer.writerow(row)

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='orange')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['acc'], color='blue')
    plt.plot(history.history['val_acc'], color='orange')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if plot:
        plt.savefig(plotFile, bbox_inches='tight')


    if export:
        json_model = model.to_json()
        with open("Model\\" + modelFile, 'w') as file:
            file.write(json_model)
        model.save_weights("Model\\" + weightFile)

plt.show()
with open(resultFile, 'w') as file:
    writer = csv.writer(file)
    for row in seqtest:
        writer.writerow(row)

ml.sum_print(starttime, repeats, seqtest)

#pyplot.plot(history['train'], color='blue')
#pyplot.plot(history['test'], color='orange')
#print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
