import numpy as np
import os, sys
from scipy.stats import zscore
from pandas import read_csv
from architecturefns import *
from collections import OrderedDict
import catch22
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from pandas import read_csv

from plotty import *
from experimentsetup import *
from preprocessingandrecords import *


def dictprecisCLA(dataset,labels, DICTYPE):
    print("DICTYPE: {}".format(DICTYPE))
    NUMPAT = 8
    CYCLELEN = 24
    # YEH + PRECIS
    ePRECIS = Experiment("PRECIS", [NUMPAT,CYCLELEN])
    ROUNDS = 1
    if DICTYPE=="RAND":
        ROUNDS = 10
    accrates = []
    for R in range(ROUNDS):
        use_dicts = []
        numdicts = 0
        for i in tqdm(range(len(dataset)),desc="Building Dictionaries"):
            ts = zscore(dataset[i],nan_policy="omit")
            d = None
            if DICTYPE == "YEH":
                d, _ = ePRECIS.make_exemplar(ts)
            elif DICTYPE == "RAND":
                d, _ = randdict(ts,NUMPAT,CYCLELEN)
            use_dicts.append(d)
            numdicts += 1

        PRECISdistmat = ePRECIS.distmat_from_dicts(use_dicts)
        accrate, _ = CLA_inference(PRECISdistmat,labels,numdicts)
        print("{}+PRECIS accrate: {}".format(DICTYPE,accrate))
        accrates.append(accrate)
    print("Average accrate over {} rounds: {}".format(ROUNDS,np.mean(accrates)))
    return

        
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


if __name__ == "__main__":
    main_dir = "/wewalksubset_accelerometer" # allsites as potential dataset?
    dataset = []
    item = 0
    fref = OrderedDict() # DISTMAT IDX
    label_colors = OrderedDict()
    idtoidx = OrderedDict()
    labels = []
    lens = []
    for (root,dirs,files) in os.walk(main_dir):
        print(len(files))
        for f in files:
            d = read_csv(main_dir+'/'+f,sep=",",header=None)
            d = d.iloc[:,3] # AccelerationTimeStamp, timeStartingAt0, Acceleration-X, Acceleration-Y, Acceleration-Z, RMSValueOFXYZ
            dataset.append(np.asarray(d)) # all rows, second column
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

    # values,counts =np.unique(lens,return_counts=True)
    # for i in zip(values,counts):
    #     print(i)

    
    catch22CLA(dataset,labels,"ALL")
    catch22CLA(dataset,labels,"BEST")
    # badidxs = catch22CLA(dataset,labels,"FS")
    # print("Catch22 Found NanVfs: {}".format(badidxs))

    # dictprecisCLA(dataset,labels,"YEH")
    # dictprecisCLA(dataset,labels,"RAND")