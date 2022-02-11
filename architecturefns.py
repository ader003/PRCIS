import numpy as np
from preprocessingandrecords import *
from modelfns import *
from dtaidistance import dtw
from scipy.stats import zscore
import mass_ts as mts
import warnings
warnings.filterwarnings('ignore')


def circ_dist_ED(w1,w2): # requires len(w1) == len(w2) 
    w1 = zscore(w1)
    w2 = zscore(w2) 
    if len(w1) != len(w2):
        print("Elements 1 and 2 are not the same length.")
        quit()
    else:
        return np.linalg.norm(w1-w2)


def circ_dist_PHASEINV(w1,w2): # this is PRECIS; the function name was an old working title during development
    w1 = zscore(w1)
    w2 = zscore(w2)
    query, longer = tq(w1,w2)
    ts = np.concatenate((longer,longer), axis=None)
    p2s = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    lb = len(query)
    ub = len(ts)
    for p in p2s:
        if p >= lb and p <= ub:
            break
    else:
        p = lb
    if len(ts) - p + 1 < 1:
        print(len(ts), len(query), p)
        print("WARNING: p is larger than the length of ts !!")
        quit()
    dp = mts.mass3(ts,query,p)
    return min(abs(dp))


def circ_dist_DTW(w1,w2):
    w1 = zscore(w1)
    w2 = zscore(w2)
    if len(w1) != len(w2):
        print("Elements 1 and 2 have unequal lengths of {} and {}.".format(len(w1),len(w2)))
        quit()
    else:
        return dtw.distance(w1,w2,use_c=True)

def listtodict(l): # list of lists
    retdict = {}
    for k in range(len(l)):
        retdict[k] = l[k]
    return retdict


def dict_dist(meta_d1, meta_d2, circ_dist, multivariate=False): # circ_dist is one of the 3 functions above
    GLOBAL_MAX = 0
    def dictcomp(d1,d2,circ_dist):
        median_list = []
        for w1 in range(len(d1)):
            nn_dist = np.inf
            for w2 in range(len(d2)):
                w2_dist = circ_dist(d1[w1],d2[w2])
                if w2_dist < nn_dist:
                    nn_dist = w2_dist
            median_list.append(nn_dist)
        return median_list

    def invoke_dictcomp(d1,d2, circ_dist):
        d1tod2 = dictcomp(d1,d2,circ_dist)
        d2tod1 = dictcomp(d2,d1,circ_dist)
        return np.concatenate((d1tod2,d2tod1))
        
    # multivar input: use_dicts: list, use_dicts[0] = dictionary with three keys; each key value is an array of lx[numsensor] ndarrays; not presented in paper and only used during development
    if multivariate: # median**2 of list of medians
        mofmedians = []
        for x in range(len(list(meta_d1.keys()))): 
            y_list = invoke_dictcomp(listtodict(meta_d1[x]),listtodict(meta_d2[x]), circ_dist) # median list
            mediany = np.median(np.asarray(y_list)) # median, pattern level
            if mediany > GLOBAL_MAX:
                GLOBAL_MAX = mediany
            mofmedians.append(mediany) 
        mofmedians = np.asarray(mofmedians)
        return np.median(mofmedians)**2 # median, dictionary level
    else:
        return np.median(dictcomp(meta_d1,meta_d2, circ_dist))**2