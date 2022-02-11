import numpy as np
from scipy.stats import zscore
from pandas import read_csv
from scipy.io import loadmat
import time
from datetime import datetime

from tqdm import tqdm
import sys, os

from experimentsetup import *
from preprocessingandrecords import *
from architecturefns import *


if __name__ == "__main__":
    args = sys.argv
    # PARAMETERS
    CYCLELEN = int(args[1])
    NUMPAT = int(args[2])
    DISTFN = args[3] #"PHASEINV"/"PRECIS" or "DTW"
    DICTYPE = args[4] # YEH, RAND, SYSRAND
    SENSOR = 1 # y acc; there are six total: xyz acc, xyz gyro
    exp = Experiment(DISTFN, [NUMPAT, CYCLELEN], algyield=True, multivariate=False)
    activitymap = {1: "Walking Forward", 2: "Walking Left", 3: "Walking Right", 4: "Walking Upstairs", 5: "Walking Downstairs", 6: "Running Forward", 7: "Jumping Up", 8: "Sitting", 9: "Standing", 10: "Sleeping", 11: "Elevator Up", 12: "Elevator Down"}
    omitactivities=["a8t","a9t","a10"]
    main_dir = 'USC-HAD/'
    i = 0
    dataset = []
    labels = []
    lens = []
    for (root,dirs,files) in os.walk(main_dir):
        for d in dirs:
            for (subj_root,subj_dirs,subj_files) in os.walk(root+d):
                for f in subj_files:
                    if f[:3] not in omitactivities: # omit sedentary
                        filedata = loadmat(main_dir+d+"/"+f)
                        filedata = np.asarray(filedata['sensor_readings'])
                        filedata = filedata[:,SENSOR] 
                        filedata = np.transpose(filedata) 
                        label = f[1:-6] 
                        lens.append(len(filedata))
                        dataset.append(zscore(filedata, nan_policy='omit', axis=0))
                        labels.append(label)
    print("FINISHED PROCESSING DATA")

    RUNLOOP = 1
    if DICTYPE=="RAND":
        RUNLOOP=10

    for ROUNDS in tqdm(range(RUNLOOP),desc="Build Dictionaries, S={}, L={}, {} {}:".format(NUMPAT,CYCLELEN, DICTYPE, DISTFN)):
        use_dicts = []
        for ts_idx in tqdm(range(len(dataset))):
            ts = dataset[ts_idx]
            d = None
            if DICTYPE=="RAND":
                d, _ = randdict(ts, NUMPAT, CYCLELEN, N_SENSORS=1)
            elif DICTYPE=="SYSRAND":
                d, _ = sysranddict(ts, NUMPAT, CYCLELEN, N_SENSORS=1)
            elif DICTYPE=="YEH":
                d, _ = exp.make_exemplar(ts)
            use_dicts.append(d)

        STARTTIME = time.time()
        distmat = exp.distmat_from_dicts(use_dicts)
        ENDTIME = time.time()
        print("FINISHED MAKING DISTMAT: {}, {}, {}, {}".format(NUMPAT, CYCLELEN, DISTFN, DICTYPE))
        accrate, errors = CLA_inference(distmat,labels,len(use_dicts))
        print("activity classification accuracy: {}".format(accrate))
