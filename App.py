import readData_AWR1642 as radar
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence
import ManipuleraData as manip
import msvcrt
from tkinter import *

testData = False
input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1"]
data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
data = data[:len(data)//100 * 100]

# ML variables set to the same as current model
batch_size = 5
time_step = 20


# Model loading data
modelFile = "model.json"
weightFile = "weights.h5"


# Prediction values
predictions = []
predLen = 10
confNumber = 6


def confidentGuess(predictions, confNumber):
    counts = {}
    for pred in predictions:
        if pred in counts:
            counts[pred] += 1
        else:
            counts[pred] = 1

    for key, val in counts.items():
        if val >= confNumber:
            return key


def loadModel(jsonFile, weightFile):
    file = open("Model\\" + jsonFile, 'r')
    loadedModelFile = file.read()
    file.close()
    load_model = model_from_json(loadedModelFile)
    load_model.load_weights("Model\\" + weightFile)
    return load_model


# init gui
root = Tk()
root.minsize(50, 100)
templabel = Label(root, text='Hey')
templabel.pack()
root.update()
update = '-'

# Main loop
mute = False
detObj = {}
key = '0'
frameData = []
frameKeys = []
currentIndex = 0
j = 0

model = loadModel(modelFile, weightFile)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# print(f' Data = {test}\n Label = {test_label}')
# print(model.predict(np.array(test).reshape(1, 10)))

'''
test = []
test_label = []
for j in range(10):
    test.append(data[40 + j][:51])
    test_label.append(40 + j)

for i in range(1):
    for j in range(10):
        test.append(data[40 + i*10 + j + 10][:51])
        test_label.append(40 + i*10 + j + 10)

    print(test)
    print(test_label)
    predict_seq = test_seq = sequence.TimeseriesGenerator(test, test_label, length=10, batch_size=10, start_index=0)
    predict = model.predict_generator(predict_seq, verbose=1)

    i = 0
    for pred in predict:
        print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])}, Confidence: {np.amax(pred)}, Actual: {test_label[i]}')
        i += 1

    test = test[10:]
    test_label = test_label[10:]
'''

i = 0
while True:
    try:
        # Update the data and check if the data is okay

        if testData:
            dataOk = True
            detObj = data[i]
            i += 1
        else:
            dataOk, detObj = radar.update(detObj)
            if dataOk:
                detObj = manip.toStandardVector(detObj)

        if dataOk:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'm':
                    mute = not mute

            # Store the current frame into frameData
            if testData:
                frameData.append(detObj[:51])
                frameKeys.append(detObj[51])
                currentIndex += 1
            else:
                frameData.append(detObj)
                frameKeys.append(key)
                currentIndex += 1

            # lastFrames.extend(frameData)
            # lastLabels.extend(frameKeys)
            #    frameData = []
            #    frameKeys = []

            if len(frameData) == time_step + 1:
                predict_seq = sequence.TimeseriesGenerator(frameData, frameKeys, length=time_step, batch_size=1)
                predict = model.predict_generator(predict_seq, verbose=1)
                frameData = frameData[1:]
                frameKeys = frameKeys[1:]

                i = 0
                if not(mute):
                    for pred in predict:
                        #print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])}, Confidence: {np.amax(pred)}, Actual: {lastLabels[i]}')
                        predictions.append(supp.int_to_label(np.where(pred == np.amax(pred))[0]))
                        while len(predictions) > predLen:
                            predictions = predictions[1:]
                        i += 1

                if update == '-':
                    update = '|'
                else:
                    update = '-'
                guess = f'{confidentGuess(predictions, confNumber)}  {update}'
                templabel.config(text=guess)
                root.update()




    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        # win.close()
        break


