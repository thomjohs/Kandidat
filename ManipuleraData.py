import csv
dataObj={}
frameData=[]
standardLen=10
cutOfIndex=40
#with open('frameData.csv') as csvfile:
#    global dataObj
#    reader = csv.DictReader(csvfile)
#    #print(reader.header())
#    for row in reader:
#        dataObj['numObj'] = row['numObj']
#        dataObj['rangeIdx'] = row['rangeIdx']
#        dataObj['dopplerIdx'] = row['dopplerIdx']
#        dataObj['peakVal'] = row['peakVal']
#        dataObj['x'] = row['x']
#        dataObj['y'] = row['y']
#        frameData.append(dataObj)

def toStandardVector(dataObj):
    global standardLen
    rangeIdx=dataObj['rangeIdx']
    mergeSort(rangeIdx)
    mappedIndexes=compareIndex(dataObj['rangeIdx'],rangeIdx)
    dataObj['rangeIdx']=rangeIdx




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

def sortOthers(dataObj,mappedIndexes):
    numberOfCutOffs=0
    numObj=dataObj['numObj']
    dopplerIdx=[]
    peakVal=[]
    x=[]
    y=[]
    #print(mappedIndexes)
    for i in mappedIndexes:
        if i<=-numObj-1:
            j=0
            #exception
        elif i<0:
            #numberOfCutOffs+=1
            #dataObj['numObj']-=1
            #del dataObj['rangeIdx'][-(i+1)]
            dopplerIdx.append(dataObj['dopplerIdx'][-(i+1)])
            peakVal.append(dataObj['peakVal'][-(i+1)])
            x.append(dataObj['x'][-(i+1)])
            y.append(dataObj['y'][-(i+1)])
        else:
            #print(numberOfCutOffs)
            dopplerIdx.append(dataObj['dopplerIdx'][i+numberOfCutOffs])
            peakVal.append(dataObj['peakVal'][i+numberOfCutOffs])
            x.append(dataObj['x'][i+numberOfCutOffs])
            y.append(dataObj['y'][i+numberOfCutOffs])
    dataObj['dopplerIdx']=dopplerIdx
    dataObj['peakVal']=peakVal
    dataObj['x']=x
    dataObj['y']=y
        
dataObj={'numObj':5,'rangeIdx':[1,12,41,6,2],'dopplerIdx':[1,2,3,4,5],'peakVal':[1,2,3,4,5],'x':[1,2,3,4,5],'y':[1,2,3,4,5],}
print(dataObj)
rangeIdx=dataObj['rangeIdx'][:]
mergeSort(rangeIdx)
mappedIndexes=compareIndex(dataObj['rangeIdx'],rangeIdx)
dataObj['rangeIdx']=rangeIdx
sortOthers(dataObj,mappedIndexes)
print(dataObj)

