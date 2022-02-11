import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist

from tqdm import tqdm

from architecturefns import *
from plotty import *
from preprocessingandrecords import *
from modelfns import *


def make_exemplar_subroutine(self,ts): # univariate, definitive
    model = MatproDict(self.cyclelen, ctx_factor=None, n_pattern=self.numpatt, save_factor=0.9, n_job=4, verbose=False) # one pattern
    model.fit(ts)
    dictionary, d_idxs = model.get_pattern()
    ret_dict, idxs = [], []
    for i in range(len(dictionary)):
        candidate = dictionary[i]
        candidate_idxs = d_idxs[i]
        if len(candidate) != self.cyclelen:
            if self.algyield==False:
                break
        ret_dict.append(candidate)
        idxs.append(candidate_idxs)
        if len(ret_dict) >= self.numpatt:
            return ret_dict, idxs
    return ret_dict, idxs


# def listtodict_subroutine(self, l): # list of lists
#     if len(l) < 1:
#         print("listtodict fn: Dictionary has no elements.")
#     retdict = {}
#     for k in range(len(l)):
#         retdict[k] = l[k]
#     return retdict


class Experiment:
    def __init__(self, distmet, dict_settings, algyield=True, multivariate=False, downsamplefactor=1):
        self.distmet = distmet
        self.numpatt = dict_settings[0]
        self.cyclelen = dict_settings[1]
        self.algyield = algyield # yield to preference of dictionary method or to exclude any generated patterns not of this exact length
        self.multivariate = multivariate
        self.downsamplefactor = downsamplefactor #typically untouched and was only used for sanity checking during development of this work


    def distmat_from_dicts(self, use_dicts): # LIST OF DICTIONARIES; EACH KEY IS A SENSOR, WHERE ITS VALUE IS A LIST OF NDARRAYS
        NUM_DICTS = len(use_dicts)
        distmat = np.zeros((NUM_DICTS, NUM_DICTS))
        for d1 in tqdm(range(NUM_DICTS), desc="Make Distmat"):
            for d2 in range(d1,NUM_DICTS):
                d = None
                if self.distmet == "ED":
                    d = dict_dist(use_dicts[d1],use_dicts[d2], circ_dist_ED, multivariate=self.multivariate) # uni, use_dicts[] is a dict k:dict(list), multi use_dicts[] k:dict(list(list))
                elif self.distmet == "DTW":
                    d = dict_dist(use_dicts[d1],use_dicts[d2], circ_dist_DTW, multivariate=self.multivariate)
                elif self.distmet=="PRECIS" or self.distmet=="PHASEINV":
                    d = dict_dist(use_dicts[d1],use_dicts[d2], circ_dist_PHASEINV, multivariate=self.multivariate)
                else:
                    print("ERROR: Out of scope distance measure.")
                    quit()
                distmat[d1,d2] = d
                distmat[d2,d1] = d
        return distmat


    def make_exemplar(self, ts): # TAKES IN A TIME SERIES, CAN BE MULTIVARIATE: A LIST OF NDARRAYS; UNIVARIATE: NDARRAY
        if self.multivariate:
            mdict, midxs = {}, {}
            for uts_idx in range(ts.shape[0]):
                uts = ts[uts_idx,:] 
                # print(uts.shape)
                d, idxs = make_exemplar_subroutine(self,uts)
                mdict[uts_idx] = d
                midxs[uts_idx] = idxs
            return mdict, midxs # RETURN DICTIONARY; VALS ARE A LIST OF NDARRAYS; KEY IS SENSOR
        else:
            d, idxs = make_exemplar_subroutine(self,ts)
            return d, idxs # LIST OF NDARRAYS


    # def listtodict(self,l): # LIST OF LIST of NDARRAYS if multivar; univar: list of NDARRAYS
    #     if self.multivariate: # feed list of dicts whose values are lists
    #         ret = []
    #         for d in range(len(l)):
    #             ret.append(listtodict_subroutine(self,l[d])) # RETURN A DICT of NDARRAYS
    #     else: # feed list whose values are a list
    #         ret = listtodict_subroutine(self,l)
    #     return ret


