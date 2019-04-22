import csv
import numpy as np
import supp
import ML_functions as ml
import random
import os
#dataObj={}
#frameData=[]
standardLen=10
cutOfIndex=40

in_file = 'PreprocessedData\\ArenSlideUp2.csv'
out_file = 'ProcessedData\\ArenSlideUp2.csv'


def file_to_frames(csv_file):
    with open(csv_file) as csvfile:
        dataList = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            data = {}
            data['numObj'] = float(row['numObj'])
            data['rangeIdx'] = supp.dString_to_farray(row, 'rangeIdx')
            data['dopplerIdx'] = supp.dString_to_farray(row, 'dopplerIdx')
            data['peakVal'] = supp.dString_to_iarray(row, 'peakVal')
            data['x'] = supp.dString_to_farray(row, 'x')
            data['y'] = supp.dString_to_farray(row, 'y')
            data['Label'] = row['Label']
            standardVec=toStandardVector(data)
            dataList.append(standardVec)
            
        #print(dataList[0][1:11])
        return dataList

def framedata_to_file(frameData,out_path):
    dataList = []
    for frame in frameData:
        standardVector = toStandardVector(frame)
        dataList.append(standardVector)
    frames_to_file(dataList, out_path)


def translate_data(data, dx, dy):
    global standardLen
    if type(data) == list:
        frameList = data[:]
        frameMatrix = np.zeros((len(frameList),52))
        i=0
        for frame in frameList:
            frameMatrix[i]=np.asarray(frame)
            i+=1
    elif ((type(data)==np.ndarray)):
        frameMatrix=data[:]
    numMatrix=frameMatrix[:,:1]
    dopplerPeakMatrix = frameMatrix[:,1+standardLen*1:standardLen*3+1]
    xMatrix = frameMatrix[:,1+standardLen*3:standardLen*4+1]
    yMatrix = frameMatrix[:,1+standardLen*4:standardLen*5+1]
    labelMatrix=frameMatrix[:,standardLen*5+1:]

    xMatrix_trans = dx*xMatrix
    yMatrix_trans = dy*yMatrix

    dx_sqr = (xMatrix_trans) ** 2
    dy_sqr = (yMatrix_trans) ** 2

    trans_xy_rangeMatrix=np.sqrt(dx_sqr + dy_sqr)

    newFrameList=np.hstack([numMatrix,trans_xy_rangeMatrix,dopplerPeakMatrix,xMatrix_trans,yMatrix_trans,labelMatrix])
    return newFrameList.tolist()

def frames_to_file(frames, out_path):
    with open(out_path, 'w', newline='') as out:
        writer = csv.writer(out)
        for frame in frames:
            writer.writerow(frame)


def toStandardVector(dataObj):
    global standardLen
    # returns copy and not reference
    rangeIdx = dataObj['rangeIdx'][:]
    mergeSort(rangeIdx)
    # mappes where the range idexes lands after sorting
    mappedIndexes = compareIndex(dataObj['rangeIdx'], rangeIdx)
    # output is at most standardLen but can be shorter
    sortOthersAndCutOf(dataObj, mappedIndexes)

    #padding
    numObj=dataObj['numObj']
    skip=0
    if numObj<standardLen:
        skip=standardLen-numObj
    if 'Label' in dataObj.keys():
        standardVector=[0]*(2+standardLen*5)
    else:
        standardVector = [0] * (1 + standardLen * 5)
    currentIndex=0
    standardVector[currentIndex]=numObj
    currentIndex+=1
    for r in dataObj['rangeIdx']:
        #print(r)
        standardVector[currentIndex]=r
        currentIndex+=1
    currentIndex+=skip
    for d in dataObj['dopplerIdx']:
        standardVector[currentIndex]=d
        currentIndex+=1
    currentIndex+=skip
    for p in dataObj['peakVal']:
        standardVector[currentIndex]=p
        currentIndex+=1
    currentIndex+=skip    
    for x in dataObj['x']:
        standardVector[currentIndex]=x
        currentIndex+=1
    currentIndex+=skip    
    for y in dataObj['y']:
        standardVector[currentIndex]=y
        currentIndex+=1

    if 'Label' in dataObj.keys():
        currentIndex += skip
        standardVector[currentIndex]=dataObj['Label']
    #print(standardVector[1:11])
    return standardVector

def mergeSort(alist):
    #print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    #print("Merging ",alist)

def compareIndex(alist,alist_sorted):
    mappedIndexes = {}
    mappedIndexesList=[]
    #for i in alist_sorted:
    #    if not mappedIndex.getKey(i) is None:
    #        mappedIndexes[i]=[]
    for i in alist:
        indexes=mappedIndexes.get(i)
        if indexes is None:
            #index=int(binaryIndexSearch(alist_sorted, i))
            for j in range(len(alist_sorted)):
                if i==alist_sorted[j]:
                    mappedIndexes[i]=[j]
                    mappedIndexesList.append(j)
                    break
                #else:
                    #print("inte med")
            
        else:
            index=indexes[len(indexes)-1]+1
            
            if i==alist_sorted[index]:
                indexes.append(index)
                mappedIndexesList.append(index)
            else:
                print("Fel")
                print(indexes)
                print(index)
                print(alist_sorted)
        
        
    return mappedIndexesList

def binaryIndexSearch(asortedlist, i):
    first=0
    last=len(asortedlist)
    while first<=last:
        midpoint=int((first+last)/2)
        currentValue=asortedlist[midpoint]
        if currentValue==i:
            return midpoint
        elif currentValue>i:
            last=midpoint-1
        else:
            first=midpoint+1
    return -1

def sortOthersAndCutOf(dataObj,mappedIndexes):
    global standardLen
    global cutOfIndex
    numObj=int(dataObj['numObj'])
    dopplerIdx = [0] * numObj
    peakVal = [0] * numObj
    x = [0] * numObj
    y = [0] * numObj
    j=0
    #print(mappedIndexes)
    for i in mappedIndexes:
        if i<0:
            a=0
            #exception
        else:
            dopplerIdx[i]=dataObj['dopplerIdx'][j]
            peakVal[i]=dataObj['peakVal'][j]
            x[i]=dataObj['x'][j]
            y[i]=dataObj['y'][j]
            j+=1
    dataObj['dopplerIdx']=dopplerIdx
    dataObj['peakVal']=peakVal
    dataObj['x']=x
    dataObj['y']=y
    #CutOf
    if numObj>standardLen:
        dataObj['rangeIdx']=dataObj['rangeIdx'][:standardLen]
        dataObj['dopplerIdx']=dataObj['dopplerIdx'][:standardLen]
        dataObj['peakVal']=dataObj['peakVal'][:standardLen]
        dataObj['x']=dataObj['x'][:standardLen]
        dataObj['y']=dataObj['y'][:standardLen]
    r=np.array(dataObj['rangeIdx'])
    b=1*(r<cutOfIndex)
    numRemaningObj=np.sum(b)
    dataObj['numObj']=numRemaningObj
    dataObj['rangeIdx']=dataObj['rangeIdx'][:numRemaningObj]
    dataObj['dopplerIdx']=dataObj['dopplerIdx'][:numRemaningObj]
    dataObj['peakVal']=dataObj['peakVal'][:numRemaningObj]
    dataObj['x']=dataObj['x'][:numRemaningObj]
    dataObj['y']=dataObj['y'][:numRemaningObj]

def translateFile(in_file, out_file):
    data=ml.load_data(in_file)
    dx=random.random()*1.5+0.5
    dy=random.random()*1.5+0.5
    if dx>2 or dx<0.5 or dy>2 or dy<0.5 :
        print(f'dx: {dx} dy: {dy}')
    data_trans=translate_data(data,dx,dy)
    data_trans_str=[]
    for frame in data_trans:
        frame_str=[str(i) for i in frame]
        data_trans_str.append(frame_str)
    frames_to_file(data_trans_str, out_file)

def translateFiles(in_files, out_files):
    for i in range(len(in_files)):
        print(f'ProcessedData\\{in_files[i]}')
        print(f'TranslatedData\\{out_files[i]}')
        translateFile(in_files[i],f'TranslatedData\\{out_files[i]}')
        
def translateFolder(input_folder,output_folder):
    for root, dirs, files in os.walk(input_folder):
        if '.csv' in files:
            files.remove('.csv')
        print(files)
        translateFiles(files,files)
def processFile(in_file, out_file):
    data=file_to_frames(in_file)
    #print(data[0][1:11])
    frames_to_file(data, out_file)

def processFiles(in_files, out_files):
    for i in range(len(in_files)):
        print(f'PreprocessedData\\{in_files[i]}')
        print(f'ProcessedData\\{out_files[i]}')
        processFile(f'PreprocessedData\\{in_files[i]}',f'ProcessedData\\{out_files[i]}')
def processFiles_folder(input_folder,output_folder):
    for root, dirs, files in os.walk(input_folder):
        if '.csv' in files:
            files.remove('.csv')
        print(files)
        processFiles(files,files)

    


# Main for testing
# dataObj={'numObj':5,'rangeIdx':[1,12,41,6,2],'dopplerIdx':[1,2,3,4,5],'peakVal':[1,2,3,4,5],'x':[1,2,3,4,5],'y':[1,2,3,4,5],}
# print(dataObj)
def main():
    frameData = file_to_frames(in_file)
    frames_to_file(frameData, out_file)

# main()
# Synthesize data
#test=[[1,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,1,1,1,1,0,0,0,0,0,0,2.5,2.5,2.5,2.5,0,0,0,0,0,0,100],[1,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,1,1,1,1,1,1,1,0,0,0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,0,0,0,100],[1,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,1,1,1,1,1,1,1,1,1,0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,0,100],[1,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,1,1,1,1,1,1,1,1,0,0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,0,0,100]]
#test=np.ones((4,52))
#print(type(test))
#in_file='PreprocessedData\\JohanButton1.csv'   
#out_file='ProcessedData\\JohanButton1.csv'
#processFiles_folder('PreprocessedData','ProcessedData')
#translateFolder('ProcessedData','TranslatedData')
#for root, dirs, files in os.walk('ProcessedData'):
#    input_button = []
#    input_swipenext = []
#    input_swipeprev = []
#    input_slideup = []
#    input_slidedown = []
#    input_flop = []
#    if '.csv' in files:
#        files.remove('.csv')
#    for file in files:
#        if 'Button' in file:
#            input_button.append(file)
#        elif 'SwipeNext' in file:
#            input_swipenext.append(file)
#        elif 'SwipePrev' in file:
#            input_swipeprev.append(file)
#        elif 'SlideUp' in file:
#            input_slideup.append(file)
#        elif 'SlideDown' in file:
#            input_slidedown.append(file)
#        elif 'Flop' in file:
#            input_flop.append(file)
#    print(input_button)
#    print(input_swipenext)
#    print(input_swipeprev)
#    print(input_slideup)
#    print(input_slidedown)
#    print(input_flop)
#    print(files)
    #processFiles(files,files)
# Translate
#in_file='JohanButton1'   
#out_file=f'TranslatedData\\JohanButton1.csv'
#translateFile(in_file,out_file)
#print(translate_data(test,2,1.5))



# Mirror some gestures

# Random noise

