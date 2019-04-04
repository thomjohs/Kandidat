import readData_AWR1642 as radar
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence
import ManipuleraData as manip
import msvcrt
from tkinter import *
from pynput.keyboard import KeyCode, Controller

testData = False
input_files = ["JohanButton1", "JohanSlideUp1", "JohanSwipeNext1",
               "ArenButton1", "ArenSlideUp1", "ArenSwipeNext1"]
data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
data = data[:len(data)//100 * 100]

# ML variables set to the same as current model
batch_size = 5
time_step = 5


# Model loading data
modelFile = "Test_1_model.json"
weightFile = "Test_1_weights.h5"


# Prediction values
predictions = []
predLen = 10
confNumber = 5


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

# init control
keyboard = Controller()
VK_volume_up = KeyCode.from_vk(0xAF)
VK_next = KeyCode.from_vk(0xB0)
Vk_play_pause = KeyCode.from_vk(0xB3)
volume = 0


# Main loop
mute = False
detObj = {}
key = '0'
frameData = []
frameKeys = []
currentIndex = 0
i = 0

model = loadModel(modelFile, weightFile)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

swiped = False
button = False
slide = False

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
                if not mute:
                    for pred in predict:
                        # print(f'Prediction: {supp.int_to_label(np.where(pred == np.amax(pred))[0])},
                        #                       Confidence: {np.amax(pred)}, Actual: {lastLabels[i]}')
                        predictions.append(supp.int_to_label(np.where(pred == np.amax(pred))[0]))
                        while len(predictions) > predLen:
                            predictions = predictions[1:]
                        i += 1

                    if update == '-':
                        update = '|'
                    else:
                        update = '-'
                    guess = confidentGuess(predictions, confNumber)
                    templabel.config(text=f'{guess} {update}')
                    root.update()

                    if guess == 'swipeNext' and not swiped:
                        swiped = True
                        print("skip")
                        keyboard.press(VK_next)
                        keyboard.release(VK_next)
                    elif guess != 'swipeNext':
                        swiped = False

                    if guess == 'button' and not button:
                        button = True
                        print('click')
                        keyboard.press(Vk_play_pause)
                        keyboard.release(Vk_play_pause)
                    elif guess != 'button':
                        button = False

                    if guess == 'slideUp':
                        if volume < 10:
                            keyboard.press(VK_volume_up)
                            keyboard.release(VK_volume_up)
                            volume += 1

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        # win.close()
        break


