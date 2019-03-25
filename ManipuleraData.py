import csv
import numpy as np
import supp

#dataObj={}
#frameData=[]
standardLen=10
cutOfIndex=40

in_file = 'frameData.csv'
out_file = 'standardData.csv'



def file_to_framedata(csv_file):
    with open(csv_file) as csvfile:

        dataList = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            data = {}
            data['numObj'] = int(row['numObj'])
            data['rangeIdx'] = supp.dString_to_iarray(row, 'rangeIdx')
            data['dopplerIdx'] = supp.dString_to_iarray(row, 'dopplerIdx')
            data['peakVal'] = supp.dString_to_iarray(row, 'peakVal')
            data['x'] = supp.dString_to_farray(row, 'x')
            data['y'] = supp.dString_to_farray(row, 'y')
            dataList.append(data)
        return dataList

def toStandardVector(dataObj):
    global standardLen
    #returns copy and not reference
    rangeIdx=dataObj['rangeIdx'][:]
    mergeSort(rangeIdx)
    #mappes where the range idexes lands after sorting
    mappedIndexes=compareIndex(dataObj['rangeIdx'],rangeIdx)
    dataObj['rangeIdx']=rangeIdx
    #output is at most standardLen but can be shorter
    sortOthersAndCutOf(dataObj, mappedIndexes)

    #padding
    numObj=dataObj['numObj']
    skip=0
    if numObj<standardLen:
        #skip the amount of zeros which is supposed to be added
        skip=standardLen-numObj-1
    standardVector=[0]*(1+standardLen*5)
    currentIndex=0
    standardVector[currentIndex]=numObj
    currentIndex+=1
    for r in dataObj['rangeIdx']:
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
    currentIndex+=skip
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
    mappedIndexes=[]
    for i in alist:
        index = binaryIndexSearch(alist_sorted, i)

        # If the object is to far away, returns the negativ index minus 1
        #if i>cutOfIndex:
        #    mappedIndexes.append(-index-1)
        # Maps the change of index when sorted
        #else:
        #   mappedIndexes.append(index)
        mappedIndexes.append(index)
        
    return mappedIndexes

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
    numberOfCutOffs=0
    numObj=dataObj['numObj']
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
            #print(numberOfCutOffs)
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
    r=np.asarray(dataObj['rangeIdx'])
    d=np.asarray(dataObj['dopplerIdx'])
    p=np.asarray(dataObj['peakVal'])
    x=np.asarray(dataObj['x'])
    y=np.asarray(dataObj['y'])
    b=1*(r<cutOfIndex)
    dataObj['numObj']=np.sum(b)
    dataObj['rangeIdx']=(r*b).tolist()
    dataObj['dopplerIdx']=(d*b).tolist()
    dataObj['peakVal']=(p*b).tolist()
    dataObj['x']=(x*b).tolist()
    dataObj['y']=(y*b).tolist()


# Main for testing
# dataObj={'numObj':5,'rangeIdx':[1,12,41,6,2],'dopplerIdx':[1,2,3,4,5],'peakVal':[1,2,3,4,5],'x':[1,2,3,4,5],'y':[1,2,3,4,5],}
# print(dataObj)

frameData = file_to_framedata(in_file)
print(frameData[0])

standardVector = toStandardVector(frameData[0])
# print(dataObj)
print(standardVector)
print(len(standardVector))


# Synthesize data

# Translate

# Mirror some gestures

# Random noise

