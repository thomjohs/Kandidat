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
import matplotlib.pyplot as plt

testData = True
if testData:
    input_file="ValideringJohan10st.csv"
    #input_files = ["JohanButton1.csv", "JohanSwipeNext1.csv",
    #           "ArenButton1.csv", "ArenSwipeNext1.csv", "GoodBackground1.csv"]
    data = ml.load_data(input_file)
    #data = supp.shuffle_gestures(ml.load_data_multiple(input_files))
    data = data[:len(data)//10 * 10]
else:
    import readData_AWR1642 as radar
    data=[0]*10

# ML variables set to the same as current model
batch_size = 10
time_step = 10
lstm_output=10

# Model loading data
modelFile = "final.json"
weightFile = "final.h5"


# Prediction values
predictions = []
predictionWindow = []
predLen = 8
confNumber = 4
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
VK_volume_down = KeyCode.from_vk(0xAE)
VK_next = KeyCode.from_vk(0xB0)
VK_prev = KeyCode.from_vk(0xB1)
Vk_play_pause = KeyCode.from_vk(0xB3)
volume = 0


# Main loop
mute = False
detObj = {}
key = '0'
frameData = []
frameKeys = []
predict = []
currentIndex = 0
i = 0

#model = loadModel(modelFile, weightFile)
#model.compile(loss='categorical_crossentropy',
              #optimizer='adam',
              #metrics=['accuracy'])



model = ml.build_lstm_single_predict(time_steps=0, vector_size=52, outputs=4, batch_size=batch_size, lstm_output=lstm_output, stateful=True)
model.load_weights(f"Model\\{weightFile}")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


temp, means, maxs = ml.load_zero_mean_normalize_data_folder("ProcessedData")

swiped = False
button = False
slide = False
tic=0
j=0
r=0
r_button=[]
r_next=[]
r_prev=[]
if testData:
    prediction_data=np.zeros([1,8])
while i<len(data):
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

            if len(frameData) == batch_size:
                #predict_seq = sequence.TimeseriesGenerator(frameData, frameKeys, length=1, batch_size=10)
                predict_seq=np.asarray(frameData)
                predict = model.predict(predict_seq.reshape(-1,1,51))
                #print(frameKeys)
                if testData:
                    label=int(frameKeys[len(frameKeys)-1])
                frameData = frameData[1:]
                frameKeys = frameKeys[1:]
                

                if not mute:
                    predict1 = np.argmax(predict, axis=1)
                    #predictions.extend(list(map(supp.int_to_label,predict1)))
                    for pred in predict1:
                        predictions.append(supp.int_to_label(pred))
                        while len(predictions) > predLen:
                            predictions = predictions[1:]
                        #print(predictions)    


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
                        if swiped:
                            j=j+1
                            if j>7:
                                swiped = False
                                j = 0
                        elif finalGuess == 'swipeNext':
                            swiped = True
                            j=0
                            r_next.append(r)
                            print('skipNext at ', r)
                            keyboard.press(VK_next)
                            keyboard.release(VK_next)
                        elif finalGuess == 'swipePrev':
                            swiped = True
                            j=0
                            r_prev.append(r)
                            print('skipPrev at ', r)
                            keyboard.press(VK_prev)
                            keyboard.release(VK_prev)
                        elif finalGuess == 'button' and not button:
                            #button = True
                            swiped = True
                            j=0
                            r_button.append(r)
                            print('click at ', r)
                            keyboard.press(Vk_play_pause)
                            keyboard.release(Vk_play_pause)
                        elif finalGuess != 'button':
                            button = False
                if testData:
                    tmp=np.array([ml.label_to_array(label,0),ml.label_to_array(label,1),ml.label_to_array(label,2),ml.label_to_array(label,7)])
                    tmp=np.hstack((tmp, predict[len(predict)-1]))
                    prediction_data=np.vstack((prediction_data,tmp))
                    r += 1
    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        # win.close()
        break
if testData:
    start = 50
    stop = len(prediction_data)
    length = stop-start
    mid = int((stop-start)/2)
    k = np.arange(len(prediction_data))

    r_button = np.asarray(r_button)
    r_next = np.asarray(r_next)
    r_prev = np.asarray(r_prev)

    r_button_sub1 = r_button[np.nonzero(np.multiply(r_button<=mid,r_button>=start))]
    r_button_sub2 = r_button[np.nonzero(np.multiply(r_button>mid,r_button<length))]
    hit_button_sub1 = np.asarray([1]*len(r_button_sub1))
    hit_button_sub2 = np.asarray([1]*len(r_button_sub2))

    r_next_sub1 = r_next[np.nonzero(np.multiply(r_next<=mid,r_next>=start))]
    r_next_sub2 = r_next[np.nonzero(np.multiply(r_next>mid,r_next<length))]
    hit_next_sub1 = np.asarray([1]*len(r_next_sub1))
    hit_next_sub2 = np.asarray([1]*len(r_next_sub2))

    r_prev_sub1 = r_prev[np.nonzero(np.multiply(r_prev<=mid,r_prev>=start))]
    r_prev_sub2 = r_prev[np.nonzero(np.multiply(r_prev>mid,r_prev<length))]
    hit_prev_sub1 = np.asarray([1]*len(r_prev_sub1))
    hit_prev_sub2 = np.asarray([1]*len(r_prev_sub2))
    #plt.subplot(2, 1, 1)
    #plt.plot(k,prediction_data[start:start+length,0],"--", color='blue')
    #plt.plot(k,prediction_data[start:start+length,1],"--", color='orange')
    #plt.plot(k,prediction_data[start:start+length,2],"--", color='red')
    #plt.plot(prediction_data[start:start+length], color='green')
    #plt.title('Input frame')
    #plt.ylabel('signal')
    #plt.xlabel('time')
    #plt.legend(['button', 'swipe next', 'swipe prev'], loc='upper left')

    #plt.subplot(2, 1, 2)
    a4 = prediction_data[:,4]>prediction_data[:,7]
    a5 = prediction_data[:,5]>prediction_data[:,7]
    a6 = prediction_data[:,6]>prediction_data[:,7]

    b4 = np.multiply(a4,prediction_data[:,4])
    b5 = np.multiply(a5,prediction_data[:,5])
    b6 = np.multiply(a6,prediction_data[:,6])

    plt.subplot(2, 1, 1)
    plt.plot(k[start:mid],prediction_data[start:mid,0],"--", color='blue')
    plt.plot(k[start:mid],prediction_data[start:mid,1],"--", color='orange')
    plt.plot(k[start:mid],prediction_data[start:mid,2],"--", color='red')

    plt.plot(k[start:mid],b4[start:mid], color='blue')
    plt.plot(k[start:mid],b5[start:mid], color='orange')
    plt.plot(k[start:mid],b6[start:mid], color='red')

    plt.plot(r_button_sub1,hit_button_sub1,'o', color='blue')
    plt.plot(r_next_sub1,hit_next_sub1,'o', color='orange')
    plt.plot(r_prev_sub1,hit_prev_sub1,'o', color='red')

    plt.title('Input frame')
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend(['button label', 'swipe next label', 'swipe prev label','button', 'swipe next', 'swipe prev'], loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(k[mid:stop],prediction_data[mid:stop,0],"--", color='blue')
    plt.plot(k[mid:stop],prediction_data[mid:stop,1],"--", color='orange')
    plt.plot(k[mid:stop],prediction_data[mid:stop,2],"--", color='red')

    plt.plot(k[mid:stop],b4[mid:stop], color='blue')
    plt.plot(k[mid:stop],b5[mid:stop], color='orange')
    plt.plot(k[mid:stop],b6[mid:stop], color='red')

    plt.plot(r_button_sub2,hit_button_sub2,'o', color='blue')
    plt.plot(r_next_sub2,hit_next_sub2,'o', color='orange')
    plt.plot(r_prev_sub2,hit_prev_sub2,'o', color='red')
    #plt.plot(k,prediction_data[start:start+length,7], color='green')
    plt.title('Input frame')
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend(['button label', 'swipe next label', 'swipe prev label','button', 'swipe next', 'swipe prev'], loc='upper left')
    plt.show()