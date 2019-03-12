def dString_to_iarray(indata, label, ):
    ilist = []
    s = ''
    for i in indata[label]:
        if str.isdigit(i) or i == '-' or i == '.':
            s += i
        elif s != '':
            ilist.append(int(s))
            s = ''
    return ilist

def dString_to_farray(indata, label, ):
    ilist = []
    s = ''
    for i in indata[label]:
        if str.isdigit(i) or i == '-' or i == '.':
            s += i
        elif s != '':
            ilist.append(float(s))
            s = ''
    return ilist