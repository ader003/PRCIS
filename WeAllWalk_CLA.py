import numpy as np
import os, sys
from scipy.stats import zscore
from pandas import read_csv
from architecturefns import *
from collections import OrderedDict
from copy import deepcopy

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from datetime import datetime
from icecream import ic
import time

from pandas import read_csv

from plotty import *
from experimentsetup import *
from preprocessingandrecords import *


def dictprecisCLA(dataset,labels, DICTYPE):
    print("DICTYPE: {}".format(DICTYPE))
    NUMPAT = 8
    CYCLELEN = 25
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

        STARTTIME = time.time()
        PRECISdistmat = ePRECIS.distmat_from_dicts(use_dicts)
        ENDTIME = time.time()

        accrate, _ = CLA_inference(PRECISdistmat,labels,numdicts)
        print("{}+PRECIS accrate: {}".format(DICTYPE,accrate))
        accrates.append(accrate)

        # log = open("icdm_logofresults.txt", "a") # append mode
        # # DATETIME,DICTYPE,DISTFN,NUMPAT,CYCLELEN,ENDTIME-STARTTIME,ACCRATE
        # log.write("{}, {}, {}, {}, {}, {}, {}\n".format(datetime.now(),DICTYPE,"PRECIS",NUMPAT,CYCLELEN,ENDTIME-STARTTIME, accrate))
        # log.close()
        
    print("Average accrate over {} rounds: {}".format(ROUNDS,np.mean(accrates)))
    return


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

    
    # catch22CLA(dataset,labels,"ALL")
    # catch22CLA(dataset,labels,"BEST")
    # badidxs = catch22CLA(dataset,labels,"FS")
    # print("Catch22 Found NanVfs: {}".format(badidxs))

    dictprecisCLA(dataset,labels,"YEH")
    dictprecisCLA(dataset,labels,"RAND")