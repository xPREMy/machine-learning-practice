import numpy as np
import matplotlib.pyplot as plt
def dw(w,b,x,y,j):
    dw=0
    for i,xi in enumerate(x):
        dw=dw+(xi[j])*((fwb(w,b,xi)-y[i]))
    return dw/x.shape[0]
def db(w,b,x,y,j):
    db=0
    for i,xi in enumerate(x):
        db=db+((fwb(w,b,xi)-y[i]))
    return db/x.shape[0]
def graddecent(w,b,x,y):
    alpha=0.2
    temp_w = np.copy(w)
    for iteratorcount in range(1000):
        print(iteratorcount)
        # in w
        comp=costf(w,b,x,y)
        for j,tw in enumerate(w):
            temp_w[j]=tw-alpha*dw(w,b,x,y,j)
        # in b
        temp_b=b-alpha*db(w,b,x,y,j)
        w=temp_w.copy()
        b=temp_b
        # for variable alpha
        if(comp<costf(w,b,x,y)):
            alpha=alpha/10
    return w,b
def fwb(w,b,x):
    return np.dot(x, w) + b
def costf(w,b,x,y):
    jwb=0
    for i,xi in enumerate(x):
        jwb=jwb+((fwb(w,b,xi)-y[i])**2)
    return jwb/(2*x.shape[0])
def pseudomain(list,x,y):
    inputlist=np.array(list)
    w=np.zeros(x.shape[1])  #  (rows,colm)
    b=1
    final_w, final_b = graddecent(w, b, x, y)
    return fwb(final_w, final_b,inputlist)
