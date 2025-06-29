import numpy as np
import math
def sigmoid(z):
     g=1/(1+np.exp(-z))
     return g
def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    total_cost=0;
    for i in range(0,m):
        total_cost=total_cost-(1-y[i])*(math.log(1-sigmoid(np.dot(w,X[i])+b)))-y[i]*math.log(sigmoid(np.dot(w,X[i])+b))
    return total_cost*(1/m)
def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    zw=np.zeros(w.shape)
    for j in range(0,n): # features 
        for i in range(0,m): 
            zw[j]=zw[j]+(sigmoid(np.dot(w,X[i])+b)-y[i])*X[i][j]
    zb=0
    for i in range(0,m):
        zb=zb+(sigmoid(np.dot(w,X[i])+b)-y[i])
    dj_db=zb*(1/m)
    dj_dw=zw*(1/m)
    return dj_db, dj_dw
def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)
    for i in range(m):   
        f_wb = sigmoid(np.dot(w,X[i])+b)
        p[i] = 1 if f_wb>=0.5 else 0
    return p
