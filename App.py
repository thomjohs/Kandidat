import readData_AWR1642 as radar
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence

input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1"]
data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
data = data[:len(data)//100 * 100]


# Configurate the serial port
#CLIport, Dataport = radar.serialConfig(radar.configFileName)

# Get the configuration parameters from the configuration file
#configParameters = radar.parseConfigFile(radar.configFileName)

# START QtAPPfor the plot
#app = radar.QtGui.QApplication([])

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
detObj = {}
frameData = []
currentIndex = 0
j = 0
lastFrames = []


model = loadModel(modelFile, weightFile)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# print(f' Data = {test}\n Label = {test_label}')
# print(model.predict(np.array(test).reshape(1, 10)))

test = []
test_label = []
for j in range(10):
    test.append(data[2040 + j][:51])
    test_label.append(2040 + j)

for i in range(3):
    for j in range(10):
        test.append(data[2040 + i*10 + j + 1][:51])
        test_label.append(2040 + i*10 + j + 1)

    predict_seq = test_seq = sequence.TimeseriesGenerator(test, test_label, length=10, batch_size=10, start_index=0)
    predict = model.predict_generator(predict_seq, verbose=1)

    i = 0
    for pred in predict:
        print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])}, Confidence: {np.amax(pred)}, Actual: {test_label[i]}')
        i += 1

    test = test[10:]
    test_label = test_label[10:]



while True:
    try:
        # Update the data and check if the data is okay
        dataOk = radar.update()

        if dataOk:
            # Store the current frame into frameData
            frameData.append(detObj)
            currentIndex += 1

        if currentIndex == 10:
            lastFrames = frameData
            frameData = []
            test_label = []

            if len(lastFrames) == 20:

                predict_seq = test_seq = sequence.TimeseriesGenerator(test, test_label, length=10, batch_size=2)
                predict = model.predict_generator(predict_seq, verbose=1)
                lastFrames = lastFrames[10:]

                i = 0
                for pred in predict:
                    print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])}, Confidence: {np.amax(pred)}, Actual: {test_label[i]}')
                    i += 1


    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        # print(frameData)
        # win.close()
        break
