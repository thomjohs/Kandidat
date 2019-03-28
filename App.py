import readData_AWR1642 as radar
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence
import ManipuleraData as manip
import msvcrt

input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1"]
data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
data = data[:len(data)//100 * 100]


# ML variables set to the same as current model

batch_size = 5


# Model loading data
modelFile = "model.json"
weightFile = "weights.h5"


def loadModel(jsonFile, weightFile):
    file = open("Model\\" + jsonFile, 'r')
    loadedModelFile = file.read()
    file.close()
    load_model = model_from_json(loadedModelFile)
    load_model.load_weights("Model\\" + weightFile)
    return load_model


# Main loop
mute = False
detObj = {}
key = '0'
frameData = []
frameKeys = []
currentIndex = 0
j = 0
lastFrames = []
lastLabels = []


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


while True:
    try:
        # Update the data and check if the data is okay
        dataOk, detObj = radar.update(detObj)

        if dataOk:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'm':
                    mute = not(mute)


            detObj = manip.toStandardVector(detObj)
            # Store the current frame into frameData
            frameData.append(detObj)
            frameKeys.append(key)
            currentIndex += 1

            if len(frameData) == 10:
                lastFrames.extend(frameData)
                lastLabels.extend(frameKeys)
                frameData = []
                frameKeys = []

                if len(lastFrames) == 20:
                    print(lastFrames)
                    predict_seq = sequence.TimeseriesGenerator(lastFrames, lastLabels, length=10, batch_size=batch_size)
                    predict = model.predict_generator(predict_seq, verbose=1)
                    lastFrames = lastFrames[10:]
                    lastLabels = lastLabels [10:]

                    i = 0
                    if not(mute):
                        for pred in predict:
                            print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])}, Confidence: {np.amax(pred)}, Actual: {lastLabels[i]}')
                            i += 1


    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        # win.close()
        break
