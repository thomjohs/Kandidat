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
    label = 7
    for frame in frameList:
        if frame[len(frame) - 1] == label:
            group.append(frame)
        else:
            if label != 7:
                if label == 6:
                    backgrounds.extend(group)
                else:
                    gestures.append(group)
            label = frame[len(frame) - 1]
            group = [frame]

    if label != 7:
        if label == 6:
            backgrounds.extend(group)
        else:
            gestures.append(group)

    while len(backgrounds) != 0 and len(gestures) != 0:
        if random.randint(0, 4) == 1:
            if len(backgrounds) < 25:
                shuffled.extend(backgrounds)
                print("LAST BACKGROUND USED!")
            else:
                randLength = random.randint(10, 25)
                i = random.randint(0, len(backgrounds)-1 - randLength)
                shuffled.extend(backgrounds[i:i + randLength])
                del backgrounds[i:i + randLength]

        i = random.randint(0, len(gestures) - 1)
        shuffled.extend(gestures[i])
        del gestures[i]

    for gesture in gestures:
        shuffled.extend(gesture)

    return shuffled


def count_gesture_length(data):
    class counter:
        current_gesture_length = 0
        n_gestures = 0
        sum_frames = 0
        max = 0
        min = float('inf')

        def __repr__(self):
            if self.n_gestures != 0:
                return f'(number of gestures:{self.n_gestures}, max: {self.max}, min: {self.min}, avg: {self.sum_frames//self.n_gestures})'
            else:
                return f'(number of gestures:{self.n_gestures}, max: {self.max}, min: {self.min}, avg: {0})'

    counters = [counter(), counter(), counter(), counter(), counter(), counter(), counter(), counter()]

    prev = 0
    for i, frame in enumerate(data):
        gesture = int(frame[len(frame) - 1])
        counters[gesture].current_gesture_length += 1

        if gesture != prev and i != 0:
            counters[prev].sum_frames += counters[prev].current_gesture_length
            counters[prev].n_gestures += 1
            if counters[prev].current_gesture_length > counters[prev].max:
                counters[prev].max = counters[prev].current_gesture_length
            if counters[prev].current_gesture_length < counters[prev].min:
                counters[prev].min = counters[prev].current_gesture_length
            counters[prev].current_gesture_length = 0

        if i == len(data) - 1:
            counters[gesture].sum_frames += counters[gesture].current_gesture_length
            counters[gesture].n_gestures += 1
            if counters[gesture].current_gesture_length > counters[gesture].max:
                counters[gesture].max = counters[gesture].current_gesture_length
            if counters[gesture].current_gesture_length < counters[gesture].min:
                counters[gesture].min = counters[gesture].current_gesture_length
            counters[gesture].current_gesture_length = 0

        prev = gesture

    return counters


def label_to_int(frame):
    last = len(frame)-1
    if frame[last] == 'slideUp' or frame[last] == '0.0':
        frame[last] = 0
    elif frame[last] == 'slideDown' or frame[last] == '1.0':
        frame[last] = 1
    elif frame[last] == 'button' or frame[last] == '2.0':
        frame[last] = 2
    elif frame[last] == 'swipeNext' or frame[last] == '3.0':
        frame[last] = 3
    elif frame[last] == 'swipePrev' or frame[last] == '4.0':
        frame[last] = 4
    elif frame[last] == 'flop' or frame[last] == '5.0':
        frame[last] = 5
    elif frame[last] == 'goodBackground' or frame[last] == '6.0':
        frame[last] = 6
    else:
        frame[last] = 7
    return frame


def int_to_label(i):
    if i == 0:
        return 'slideUp'
    elif i == 1:
        return 'slideDown'
    elif i == 2:
        return 'button'
    elif i == 3:
        return 'swipeNext'
    elif i == 4:
        return 'swipePrev'
    elif i == 5:
        return 'flop'
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



