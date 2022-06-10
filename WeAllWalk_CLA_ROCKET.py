import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket
from tqdm import tqdm
from scipy.io import loadmat
from scipy.stats import zscore
from collections import OrderedDict
import os, sys
from pandas import read_csv
from architecturefns import *

from plotty import *
from experimentsetup import *
from preprocessingandrecords import *


if __name__ == "__main__":
    main_dir = "wewalksubset_accelerometer" # allsites as potential dataset?
    dataset = []
    item = 0
    fref = OrderedDict() # DISTMAT IDX
    label_colors = OrderedDict()
    idtoidx = OrderedDict()
    labels = []
    lens = []
    for (root,dirs,files) in os.walk(main_dir):
        for f in files:
            d = read_csv(main_dir+'/'+f,sep=",",header=None)
            d = d.iloc[:,3] # AccelerationTimeStamp, timeStartingAt0, Acceleration-X, Acceleration-Y, Acceleration-Z, RMSValueOFXYZ
            dataset.append(zscore(np.asarray(d),nan_policy='omit')) # all rows, second column
            lens.append(len(d))
            txt = f[:-23]
            txt = txt.replace("2R_","")
            fref[item] = txt.replace("_"," ")
            tmp = fref[item]
            idtoidx[fref[item]] = item
            if "ID6" in fref[item]:
                label_colors[tmp] = 'r'
                labels.append("ID6")
            elif "ID7" in fref[item]:
                label_colors[tmp] = 'g'
                labels.append("ID7")
            elif "ID8" in fref[item]:
                label_colors[tmp] = 'b'
                labels.append("ID8")
            elif "ID11" in fref[item]:
                label_colors[tmp] = 'c'
                labels.append("ID11")
            elif "ID13" in fref[item]:
                label_colors[tmp] = 'm'
                labels.append("ID13")
            elif "ID14" in fref[item]:
                label_colors[tmp] = 'y'
                labels.append("ID14")
            else:
                label_colors[tmp] = 'k'
            item += 1


    # PAD
    longest_len = max(lens)
    padtype = "self" # also "self"
    for ts_idx in range(len(dataset)):
        ts = dataset[ts_idx]
        if padtype == "zero":
            dataset[ts_idx] = np.pad(ts,(0,longest_len-len(ts)),mode='constant')
        elif padtype == "self":
            dataset[ts_idx] = np.pad(ts,(0,longest_len-len(ts)),mode='wrap')


    rocket_labels_MAIN = np.asarray(labels)
    rocket_dataset_MAIN = np.stack(dataset,axis=0)
    rocket_dataset_MAIN = np.expand_dims(rocket_dataset_MAIN,axis=2)

    scores = []
    for i in tqdm(range(len(rocket_labels_MAIN))): # LOO
        X_train, X_test = np.delete(rocket_dataset_MAIN,i,axis=0), rocket_dataset_MAIN[i]
        X_test = np.expand_dims(X_test,axis=0)
        y_train, y_test = np.delete(rocket_labels_MAIN,i), rocket_labels_MAIN[i]

        X_test = np.concatenate((X_test, X_test)) # due to sktime implementation, it requires X_test to have more than one datapoint
        y_test = np.expand_dims(y_test,axis=0)
        y_test = np.concatenate((y_test,y_test))
        y_test = np.expand_dims(y_test,axis=1) # again
        
        rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
        rocket.fit(X_train)
        X_train_transform = rocket.transform(X_train)
        classifier = RidgeClassifierCV(alphas=np.logspace(-3,3,10), normalize=False) # normalized already
        classifier.fit(X_train_transform, y_train)

        X_test_transform = rocket.transform(X_test)
        scores.append(classifier.score(X_test_transform, y_test))

    print("WeAllWalk, ROCKET, {} padding, LOO Accuracy: {}".format(padtype,np.mean(scores)))


    # WeAllWalk, ROCKET, zero padding, LOO Accuracy: 0.2857142857142857