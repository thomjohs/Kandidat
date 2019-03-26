import random

def dString_to_iarray(indata, label):
    ilist = []
    s = ''
    for i in indata[label]:
        if str.isdigit(i) or i == '-' or i == '.':
            s += i
        elif s != '':
            ilist.append(int(s))
            s = ''
    return ilist


def dString_to_farray(indata, label='0'):
    ilist = []
    s = ''
    if label == '0':
        for i in indata:
            if str.isdigit(i) or i == '-' or i == '.':
                s += i
            elif s != '':
                ilist.append(float(s))
                s = ''
        return ilist
    else:
        for i in indata[label]:
            if str.isdigit(i) or i == '-' or i == '.':
                s += i
            elif s != '':
                ilist.append(float(s))
                s = ''
        return ilist

def shuffle_gestures(frameList):
    shuffled = []
    backgrounds = []
    gestures = []
    group = []
    label = 'background'
    i = 0
    for frame in frameList:
        if frame[len(frame) - 1] == label:
            group.append(frame)
        else:
            if label == 'background':
                if False:
                    backgrounds.append(group)
                i += 1
            else:
                gestures.append(group)

            label = frame[len(frame) - 1]
            group = [frame]


    while len(backgrounds) != 0 and len(gestures) != 0:
        i = random.randint(0, len(backgrounds)-1)
        shuffled.extend(backgrounds[i])
        del backgrounds[i]

        i = random.randint(0, len(gestures) - 1)
        shuffled.extend(gestures[i])
        del gestures[i]

    for gesture in gestures:
        shuffled.extend(gesture)

    for background in backgrounds:
        shuffled.extend(background)

    return shuffled


def label_to_int(frame):
    last = len(frame)-1
    if frame[last] == 'slideUp':
        frame[last] = 0
    elif frame[last] == 'button':
        frame[last] = 1
    elif frame[last] == 'swipeNext':
        frame[last] = 2
    else:
        frame[last] = 3
    return frame


def int_to_label(i):
    if i == 0:
        return 'slideUp'
    elif i == 1:
        return 'button'
    elif i == 2:
        return 'swipeNext'
    else:
        return 'background'


def noise(num):
    num += num/2*random.randint(-1, 1)
    return num


def create_gesture1(amount, vector_size):
    frameList = []
    for i in range(amount):
        frame = []
        for j in range(vector_size):
            frame.append(noise(j))
        frame.append('slideUp')
        frameList.append(frame)
    return frameList


def create_gesture2(amount, vector_size):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(noise(11))
        frame.append('button')
        frameList.append(frame)
    return frameList


def create_gesture3(amount, vector_size):
    frameList = []
    for i in range(amount):
        frame = []
        for _ in range(vector_size):
            frame.append(random.randint(0, 40))
        frame.append('background')
        frameList.append(frame)
    return frameList





def create_data(vector_size):
    frameList = []
    for i in range(135):
        randGest = random.randint(1, 2)
        randAmount = 40 + random.randint(-10, 10)
        if randGest == 1:
            frameList.extend(create_gesture1(randAmount, vector_size))
        elif randGest == 2:
            frameList.extend(create_gesture2(randAmount, vector_size))
        else:
            frameList.extend(create_gesture3(randAmount, vector_size))

    return frameList[:4000]


def create_unshuffeled(vector_size):
    frameList = []
    for i in range(135):
        randAmount = 5 # + random.randint(-2, 2)
        if i % 2 == 0:
            frameList.extend(create_gesture3(randAmount, vector_size))
        elif i < 135/2:
            frameList.extend(create_gesture1(randAmount, vector_size))
        else:
            frameList.extend(create_gesture2(randAmount, vector_size))

    return frameList[:4000]



