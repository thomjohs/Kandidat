import csv
dataObj={}
frameData=[]
with open('frameData.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    #print(reader.header())
    for row in reader:
        dataObj['numObj'] = row['numObj']
        dataObj['rangeIdx'] = row['rangeIdx']
        dataObj['dopplerIdx'] = row['dopplerIdx']
        dataObj['peakVal'] = row['peakVal']
        dataObj['x'] = row['x']
        dataObj['y'] = row['y']
        frameData.append(dataObj)

#def toStandard vector(dataObj):
#    rangeIdx=dataObj[rangeIdx]
#    mergeSort(alist)

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


def binaryIndexSearch(asortedlist, i):
    first=0
    last=len(asortedlist)
    while first<=last:
        midpoint=(first-last)/2
        currentValue=asortedlist[midpoint]
        if currentValue==i:
            return midpoint
        elif currentValue>i:
            last=midpoint-1
        else:
            first=midpoint+1
    return -1

    
binaryIndexSearch([1,2,3],2)