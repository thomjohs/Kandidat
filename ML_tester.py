import supp
import tensorflow as tf
import ML_functions as ml
import array as arr
import datetime
from keras.preprocessing import sequence
from keras.callbacks import LambdaCallback
from keras import optimizers
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


input_button = ['AlexButton1.csv', 'AlexButton5.csv', 'AndreasButton1.csv', 'AndreasButton2.csv', 'AndreasButton3.csv', 'AndreasButton4.csv', 'AndreasButton5.csv', 'ArenButton1.csv', 'ArenButton2.csv', 'ArenButton3.csv', 'JohanButton1.csv', 'JohanButton2.csv', 'JuliaButton1.csv', 'LinusButton1.csv', 'MartinButton1.csv', 'MatildaButton1.csv']

input_swipenext = ['AlexSwipeNext1.csv', 'AlexSwipeNext5.csv', 'AndreasSwipeNext1.csv', 'AndreasSwipeNext2.csv', 'AndreasSwipeNext3.csv', 'AndreasSwipeNext4.csv', 'AndreasSwipeNext5.csv', 'ArenSwipeNext1.csv', 'ArenSwipeNext2.csv', 'ArenSwipeNext3.csv', 'JohanSwipeNext1.csv', 'JohanSwipeNext2.csv', 'JuliaSwipeNext1.csv', 'LinusSwipeNext.csv', 'LinusSwipeNext1.csv', 'MartinSwipeNext1.csv', 'MatildaSwipeNext1.csv']

input_swipeprev = ['AlexSwipePrev1.csv', 'ArenSwipePrev1.csv', 'JohanSwipePrev1.csv', 'JohanSwipePrev2.csv', 'JuliaSwipePrev1.csv', 'LinusSwipePrev.csv', 'LinusSwipePrev1.csv', 'MartinSwipePrev1.csv', 'MatildaSwipePrev.csv', 'MatildaSwipePrev1.csv']

input_slideup = ['AlexSlideUp1.csv', 'AlexSlideUp5.csv', 'AndreaSlideUp1.csv', 'AndreasSlideUp1.csv', 'AndreasSlideUp2.csv', 'AndreasSlideUp3.csv', 'AndreasSlideUp4.csv', 'AndreasSlideUp5.csv', 'ArenSlideUp1.csv', 'ArenSlideUp2.csv', 'ArenSlideUp3.csv', 'JohanSlideUp2.csv', 'JuliaSlideUp1.csv', 'LindaSlideUp1.csv', 'LinusSlideUp1.csv', 'MartinSlideUp1.csv', 'MatildaSlideUp1.csv']

input_slidedown = ['AlexSlideDown1.csv', 'ArenSlideDown1.csv', 'JohanSlideDown1.csv', 'JohanSlideDown2.csv', 'JuliaSlideDown1.csv', 'JuliaSlideDown2.csv', 'LinusSlideDown1.csv', "MartinSlideDown1'.csv", 'MartinSlideDown1.csv', 'MatildaSlideDown1.csv']

input_flop = ['AlexFlop1.csv', 'ArenFlop1.csv', 'JohanFlop1.csv', 'JohanFlop2.csv', 'JuliaFlop1.csv', 'LinusFlop1.csv', 'MartinFlop1.csv', 'MatildaFlop1.csv']

input_background = ["GoodBackground1.csv", "GoodBackground2.csv"]


input_files = input_button + input_swipenext + input_background+ input_swipeprev + \
              input_slideup + input_slidedown + input_flop

input_folder = "ProcessedData"
art_folder = "TranslatedData"

# Number of categories
outputs = 7

# training hyperparameters

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

epochs = 300
time_steps = 10
batch_size = 10
learning_rate = 0.001
decay = 1/(10**1)
=======
epochs = 2000
time_steps = 10
batch_size = 10
learning_rate = 0.00001
decay = 1/(10**6)
>>>>>>> parent of 870ab52... Merge branch 'master' of https://github.com/thomjohs/Kandidat
=======
epochs = 10
=======
epochs = 2000
>>>>>>> parent of 870ab52... Merge branch 'master' of https://github.com/thomjohs/Kandidat
time_steps = 10
batch_size = 10
learning_rate = 0.00001
decay = 1/(10**6)
>>>>>>> parent of 8991593... Update ML_tester.py
=======
epochs = 10
time_steps = 10
batch_size = 10
learning_rate = 0.00025
decay = 1/(10**6)
>>>>>>> parent of 8991593... Update ML_tester.py

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
modelSaveFile = f'ts{time_steps}bs{batch_size}lstmout{lstm_output}st{stateful}lr{learning_rate}.json'
weightSaveFile = f'ts{time_steps}bs{batch_size}lstmout{lstm_output}st{stateful}lr{learning_rate}.h5'

# Model loading data
load = False
modelFile = "ts10bs10lstmout20stTruelr0.00025.json"
weightFile = "ts10bs10lstmout20stTruelr0.00025.h5"

# optimizers
# adam standard: (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optadam = optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)

# rmsprop standard: (lr=0.001, rho=0.9, epsilon=None, decay=0.0)
optprop = optimizers.rmsprop(lr=learning_rate, rho=0.9, epsilon=None, decay=decay)

runopt = optadam

# saves plot
plot = True
plotFile = f'Plots\\ts{time_steps}bs{batch_size}lstmout{lstm_output}st{stateful}lr{learning_rate}.svg'

# saves Result
resultFile = "results.csv"


data_norm, means, maxs = ml.load_zero_mean_normalize_data_folder(input_folder)

data = supp.shuffle_gestures(data_norm)

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


for i in range(repeats):

    if load:
        model = ml.loadModel(modelFile, weightFile)
    else:
        model = ml.build_lstm(time_steps, vector_size, outputs, batch_size, lstm_output, stateful)
        # model = ml.build_clstm(time_steps, vector_size, outputs, num_filters, kernel_size, lstm_output)
        # model = ml.build_crrr(time_steps, vector_size, outputs, num_filters, batch_size, kernel_size, lstm_output, stateful)

    model.compile(loss='categorical_crossentropy',
                  optimizer=runopt,
                  metrics=['accuracy'])

    history = model.fit_generator(train_seq,
                                  callbacks=[LambdaCallback(
                                      on_epoch_begin=lambda epoch, logs: print('Repeats', i + 1, '/', repeats))],
                                  epochs=epochs,
                                  validation_data=test_seq)

    seqtest.append(model.evaluate_generator(test_seq))

    predictions = model.predict_generator(test_seq)
    predictions = np.argmax(predictions, axis=1)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> parent of 8991593... Update ML_tester.py
    cm = confusion_matrix(np.argmax(y_test[time_steps:], axis=1), predictions)
    print(cm)
    print()
    print()
<<<<<<< HEAD
>>>>>>> parent of 8991593... Update ML_tester.py
=======
    cm = confusion_matrix(np.argmax(y_train[:len(y_train) // 1000 * 1000], axis=1), predictions)
    print(cm)
>>>>>>> parent of 870ab52... Merge branch 'master' of https://github.com/thomjohs/Kandidat
=======
>>>>>>> parent of 8991593... Update ML_tester.py

    cm = confusion_matrix(np.argmax(y_train[:len(y_train) // 1000 * 1000+ time_steps], axis=1), predictions)
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
        plt.tight_layout()
        plt.savefig(plotFile, format='svg')


    if export:
        json_model = model.to_json()
        with open("Model\\" + modelSaveFile, 'w') as file:
            file.write(json_model)
        model.save_weights("Model\\" + weightSaveFile)

plt.show()
with open(resultFile, 'w') as file:
    writer = csv.writer(file)
    for row in seqtest:
        writer.writerow(row)

ml.sum_print(starttime, repeats, seqtest)

#pyplot.plot(history['train'], color='blue')
#pyplot.plot(history['test'], color='orange')
#print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
