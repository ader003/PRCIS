import numpy as np
import pandas as pd
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from icecream import ic
import time
from tqdm import tqdm
from scipy.stats import zscore
from copy import deepcopy
import catch22

def catch22_ALL(featurevectors,labels,lendataset):
    # CATCH22 (ALL FEATURES)
    correct = 0
    loo = KNeighborsClassifier(n_neighbors=1, n_jobs=4) # ED
    for test_index in tqdm(range(lendataset),desc="Catch22, All Features LOO"):
        X_train, X_test = np.delete(featurevectors,test_index,axis=0), featurevectors[test_index]
        y_train, y_test = np.delete(labels,test_index), labels[test_index]
        loo.fit(X_train,y_train)
        nn_label = loo.predict([X_test])
        if nn_label == y_test:
            correct += 1
    print("Catch-22 (All Features) LOO Accuracy: {}".format(correct/lendataset))
    return


def catch22_BEST(featurevectors,labels,lendataset,featurenames):
    # BEST (FEATURE) OF CATCH22
    global_best_acc = 0
    global_best_feature = featurenames[0]
    loo = KNeighborsClassifier(n_neighbors=1, n_jobs=4) # ED
    for f in tqdm(range(22),desc="Catch22 Best Feature Run Through"):
        curr_acc = 0
        correct = 0
        data = np.expand_dims(featurevectors[:,f],axis=1)
        for test_index in tqdm(range(lendataset)):
            X_train, X_test = np.delete(data,test_index,axis=0), data[test_index]
            y_train, y_test = np.delete(labels,test_index), labels[test_index]
            loo.fit(X_train,y_train)
            nn_label = loo.predict([X_test])
            if nn_label == y_test:
                correct += 1
        curr_acc = correct/lendataset
        if curr_acc > global_best_acc:
            global_best_acc = curr_acc
            global_best_feature = featurenames[f]
    print("Catch-22 (Best of Features) LOO Accuracy, Feature: {}, {}".format(global_best_acc, global_best_feature))
    return


def catch22_FS(featurevectors, labels, lendataset, featurenames):
    # FORWARD SELECTION ON CATCH22 FEATURES
    global_best_acc = 0
    use_features = []
    lastfeature = None
    featuresbank = np.arange(22)
    featurenamesbank = featurenames
    featurenames = np.array(featurenames)
    loo = KNeighborsClassifier(n_neighbors=1, n_jobs=4) # ED
    
    while(len(featuresbank)>0):
        global_best_feature_idx = None
        global_best_feature_name = ""
        for f in tqdm(range(len(featuresbank)), desc="FS Feature Run Through:"):
            curr_acc = 0
            correct = 0
            curr_usefeatures = deepcopy(use_features)
            curr_usefeatures.append(featuresbank[f])
            currfeaturesdata = featurevectors[:,curr_usefeatures]
            for test_index in range(lendataset):
                X_train, X_test = np.delete(currfeaturesdata,test_index,axis=0), currfeaturesdata[test_index]
                y_train, y_test = np.delete(labels,test_index), labels[test_index]
                loo.fit(X_train,y_train)
                nn_label = loo.predict([X_test])
                if nn_label == y_test:
                    correct += 1
            curr_acc = correct/lendataset
            if curr_acc > global_best_acc:
                global_best_acc = curr_acc
                global_best_feature_name = featurenamesbank[f]
                global_best_feature_idx = f
        if global_best_feature_idx == None:
            print("BREAK, use_features, rejectedfeatures: {}, {}".format(use_features,featuresbank))
            break # stop adding more features
        else:
            lastfeature = featuresbank[global_best_feature_idx]
            use_features.append(lastfeature)
            featuresbank = np.delete(featuresbank,global_best_feature_idx)
            featurenamesbank = np.delete(featurenamesbank,global_best_feature_idx)
            ic(global_best_feature_name,use_features,featuresbank)
    print("Catch-22 (FS on Features) LOO Accuracy, Features: {}, {}".format(global_best_acc, np.array(featurenames[use_features])))
    return 


def catch22CLA(dataset,labels,C22TYPE):
    print("C22TYPE: {}".format(C22TYPE))
    # CATCH22
    featurevectors = []
    featurenames = []
    badidxs = []
    STARTTIME = datetime.now()
    for ts_idx in tqdm(range(len(dataset)),desc="Catch-22, Compute Feature Vectors"):
        ts = zscore(dataset[ts_idx],nan_policy="omit")
        good = ~np.isnan(ts)
        bad = np.isnan(ts)
        avg = np.mean(ts[good])
        ts[bad] = avg
        fV = catch22.catch22_all(ts)
        fV_values = fV['values']
        if np.isnan(fV_values).any():
            badidxs.append(ts_idx)
            continue
        else:
            featurevectors.append(fV_values)
    labels = np.asarray(labels)
    print("Nan TSes: {}, labels: {}".format(badidxs,labels[badidxs]))
    newlabels = np.delete(labels,badidxs)
    featurenames = fV['names'] # names are the same for all
    featurevectors = np.asarray(featurevectors)
    lendataset=len(newlabels)
    if C22TYPE=="ALL":
        catch22_ALL(featurevectors,newlabels,lendataset)
    elif C22TYPE=="BEST":
        catch22_BEST(featurevectors,newlabels,lendataset,featurenames)
    elif C22TYPE=="FS":
        catch22_FS(featurevectors, newlabels, lendataset, featurenames)
    ENDTIME = datetime.now()
    print("TIME ELAPSED, C22 FS: {}".format(ENDTIME-STARTTIME))
    return badidxs


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

def movemean(ts, smooth):
    return np.convolve(ts, np.ones(smooth)/smooth, mode='valid')

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
                if np.isnan(ts).any():
                    good = ~np.isnan(ts)
                    bad = np.isnan(ts)
                    avg = np.mean(ts[good])
                    ts[bad] = avg
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