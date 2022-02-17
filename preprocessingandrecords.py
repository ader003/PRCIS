import numpy as np
# from pandas import read_csv
# import scipy as scp
import pandas as pd
from random import randrange

def CLA_inference(distmat, labels, numdicts):
    acc_activity = 0
    preddict = {}
    errors = []
    for i in range(numdicts):
        nns = np.argsort(distmat[i])
        nn = nns[1] # ignore self
        pred = labels[nn]
        truth = labels[i]
        if (pred,truth) in preddict.keys():
            preddict[pred,truth] += 1
        else:
            preddict[pred,truth] = 1
        if pred == truth:
            acc_activity += 1
        else:
            a = [i, nn, distmat[i][nns[nn]], labels[nn], labels[i]]
            errors.append(a)
    accrate = acc_activity/numdicts
    return accrate, errors


def movemin(ts, stride):
    return pd.Series(ts).rolling(stride).min().dropna().tolist() # https://stackoverflow.com/questions/43288542/max-in-a-sliding-window-in-numpy-array

def smooth(a,WSZ): # https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number, as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


def randomwindows(ts, NUMPAT_cpy, WINLEN, N_SENSORS):
    windows = None
    if N_SENSORS > 1:
        while (WINLEN*NUMPAT_cpy) > ts.shape[1]:
            NUMPAT_cpy -= 1
        if NUMPAT_cpy > 1:
            windows = [(i*(ts.shape[1]//NUMPAT_cpy))-WINLEN+1 for i in range(1,NUMPAT_cpy)]
        else:
            windows = [ts.shape[1]-WINLEN]
    else:
        while (WINLEN*NUMPAT_cpy) > len(ts):
            NUMPAT_cpy -= 1
        if NUMPAT_cpy > 1:
            windows = [(i*(len(ts)//NUMPAT_cpy))-WINLEN+1 for i in range(1,NUMPAT_cpy)]
        else:
            windows = [len(ts)-WINLEN]
    return windows


def randdict(ts, NUMPAT, WINLEN, N_SENSORS=1):
    NUMPAT_cpy = NUMPAT
    windows = windows = randomwindows(ts, NUMPAT_cpy, WINLEN, N_SENSORS)

    def randdict_subroutine(ts,s,multivariate,windows):
        d, idxs = [], []
        windowstart = 0
        for i in range(len(windows)):
            start = randrange(windowstart,windows[i])
            end = start+WINLEN
            idxs.append((start,end))
            if multivariate:
                tmp = ts[s,start:end]
                d.append(tmp) # l x 3 if multivar
            else:
                tmp = ts[start:end]
                d.append(tmp) 
        return d,idxs

    if N_SENSORS > 1:
        dret, idxsret = {}, {}
        for s in range(N_SENSORS):
            dtmp, idxstmp = randdict_subroutine(ts,s,True,windows)
            dret[s] = dtmp
            idxsret[s] = idxstmp
        return dret, idxsret
    else:
        return randdict_subroutine(ts,0,False,windows) # second parameter is for multivariate cases


def sysranddict(ts, NUMPAT, WINLEN, N_SENSORS=1): # not used in Tables X and Y
    NUMPAT_cpy = NUMPAT
    windows = randomwindows(ts, NUMPAT_cpy, WINLEN, N_SENSORS)

    def sysranddict_subroutine(ts,s,multivariate, windows):
        d, idxs = [], []
        for i in range(len(windows)):
            start = windows[i]
            end = start+WINLEN
            idxs.append((start,end))
            if multivariate:
                d.append(ts[s,start:end]) # N_SENSORSxlen if multivar
            else:
                d.append(ts[start:end])
        return d,idxs

    if N_SENSORS > 1: #needs finessing if you want gyro and not acc on USCHAD\
        dret, idxsret = {}, {}
        for s in range(N_SENSORS):
            dtmp, idxstmp = sysranddict_subroutine(ts,s,True, windows)
            dret[s] = dtmp
            idxsret[s] = idxstmp
        return dret, idxsret
    else:
        return sysranddict_subroutine(ts,0,False, windows)



def tq(w1,w2):
    shorter, longer = None, None
    if len(w1) > len(w2):
        shorter = w2
        longer = w1
    else:
        shorter = w1
        longer = w2
    return shorter, longer