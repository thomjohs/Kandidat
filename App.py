
from keras.models import model_from_json
import supp
import ML_functions as ml
import numpy as np
from keras.preprocessing import sequence
import ManipuleraData as manip
import msvcrt
from tkinter import *
from pynput.keyboard import KeyCode, Controller
import time

testData = True
if testData:
    input_files = ["JohanButton1.csv", "JohanSwipeNext1.csv",
               "ArenButton1.csv", "ArenSwipeNext1.csv", "GoodBackground1.csv"]
    data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
    data = data[:len(data)//100 * 100]
else:
    import readData_AWR1642 as radar

# ML variables set to the same as current model
batch_size = 10
time_step = 10


# Model loading data
modelFile = "ts10bs10lstmout20stTruelr1e-05.json"
weightFile = "ts10bs10lstmout20stTruelr1e-05.h5"


# Prediction values
predictions = []
predictionWindow = []
predLen = 8
confNumber = 5
guess = 'background'
finalGuess = 'background'

# Guesses
guesses = []
guessLen = 9
confNumberGuess = 5


def confidentGuess(predictions, confNumber):
    counts = {}
    for pred in predictions:
        if pred != 'background':
            if pred in counts:
                counts[pred] += 1
            else:
                counts[pred] = 1

    for key, val in counts.items():
        if val >= confNumber:
            return key
    return 'background'


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



model = ml.build_lstm_single_predict(time_steps=time_step, vector_size=52, outputs=4, batch_size=1, lstm_output=20, stateful=True)
model.load_weights("Model\\ts10bs10lstmout20stTruelr0.0025.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


temp, means, maxs = ml.load_zero_mean_normalize_data_folder("ProcessedData")

swiped = False
button = False
slide = False
tic=0
while True:
    try:
        # Update the data and check if the data is okay

        if testData:
            dataOk = True
            detObj = ml.zero_mean_normalize_data_frame(data[i], means, maxs)
            i += 1

        else:
            dataOk, detObj = radar.update(detObj)
            if dataOk:
                detObj = manip.toStandardVector(detObj)
                detObj = ml.zero_mean_normalize_data_frame(detObj,means,maxs)


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

            if len(frameData) == time_step:
                #predict_seq = sequence.TimeseriesGenerator(frameData, frameKeys, length=1, batch_size=10)
                predict_seq=np.asarray(frameData)
                predict = model.predict(predict_seq.reshape(-1,1,51))
                frameData = frameData[1:]
                frameKeys = frameKeys[1:]

                i = 0
                if not mute:
                    predict1 = np.argmax(predict, axis=1)
                    #predictions.extend(list(map(supp.int_to_label,predict1)))
                    for pred in predict1:
                        print(supp.int_to_label(pred))
                        predictions.append(supp.int_to_label(pred))
                        while len(predictions) > predLen:
                            predictions = predictions[1:]
                            


                        if update == '-':
                            update = '|'
                        else:
                            update = '-'
                        guess = confidentGuess(predictions, confNumber)

                        guesses.append(guess)
                        while len(guesses) > guessLen:
                            #print(guesses)
                            guesses=guesses[1:]
                            finalGuess = confidentGuess(guesses, confNumberGuess)
                            #print(finalGuess)
                            if finalGuess !='background':
                                guesses = []

                            templabel.config(text=f'{finalGuess} {update}')
                            root.update()

                            if finalGuess == 'swipeNext' and not swiped:
                                swiped = True
                                print("skip")
                                #keyboard.press(VK_next)
                                #keyboard.release(VK_next)
                            elif finalGuess != 'swipeNext':
                                swiped = False

                            if finalGuess == 'button' and not button:
                                button = True
                                print('click')
                                #keyboard.press(Vk_play_pause)
                                #keyboard.release(Vk_play_pause)
                            elif finalGuess != 'button':
                                button = False

                            if finalGuess == 'slideUp':
                                if volume < 10:
                                    #keyboard.press(VK_volume_up)
                                    #keyboard.release(VK_volume_up)
                                    volume += 1

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        # win.close()
        break