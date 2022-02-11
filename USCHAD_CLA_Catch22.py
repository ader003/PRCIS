from xml.sax.handler import feature_external_ges
import numpy as np
from scipy.stats import zscore
from scipy.io import loadmat
import os

import catch22
from icecream import ic
from sklearn.neighbors import KNeighborsClassifier

from experimentsetup import *
from preprocessingandrecords import *
from architecturefns import *
from copy import deepcopy

# Results are pasted at bottom
# Uncomment blocks of code corresponding to which settings you want to subject C22 to
if __name__ == "__main__":
    SENSOR = 1
    activitymap = {1: "Walking Forward", 2: "Walking Left", 3: "Walking Right", 4: "Walking Upstairs", 5: "Walking Downstairs", 6: "Running Forward", 7: "Jumping Up", 8: "Sitting", 9: "Standing", 10: "Sleeping", 11: "Elevator Up", 12: "Elevator Down"}
    omitactivities=["a8t","a9t","a10"]
    main_dir = '/home/ader003/DATASETS/USC-HAD/'
    i = 0
    dataset = []
    labels = []
    lens = []
    for (root,dirs,files) in os.walk(main_dir):
        for d in dirs:
            for (subj_root,subj_dirs,subj_files) in os.walk(root+d):
                for f in subj_files:
                    if f[:3] not in omitactivities:
                        filedata = loadmat(main_dir+d+"/"+f)
                        filedata = np.asarray(filedata['sensor_readings'])
                        filedata = filedata[:,SENSOR] 
                        filedata = np.transpose(filedata) 
                        label = f[1:-6] 
                        lens.append(len(filedata))
                        dataset.append(zscore(filedata, nan_policy='omit', axis=0))
                        labels.append(label)
    print("FINISHED PROCESSING DATA")

    featurevectors = []
    featurenames = []
    lendataset = len(dataset)
    for ts_idx in tqdm(range(len(dataset)),desc="USCHAD Catch-22, Compute Feature Vectors"):
        ts = dataset[ts_idx]
        good = ~np.isnan(ts)
        bad = np.isnan(ts)
        avg = np.mean(ts[good])
        ts[bad] = avg
        fV = catch22.catch22_all(ts)
        fV_values = fV['values']
        featurenames = fV['names']
        featurevectors.append(fV_values)

    featurevectors = np.asarray(featurevectors)
    labels = np.asarray(labels)
    loo = KNeighborsClassifier(n_neighbors=1, n_jobs=4) # ED

    # CATCH22 (ALL FEATURES)
    # correct = 0
    # for test_index in tqdm(range(lendataset)):
    #     X_train, X_test = np.delete(featurevectors,test_index,axis=0), featurevectors[test_index]
    #     y_train, y_test = np.delete(labels,test_index), labels[test_index]
    #     loo.fit(X_train,y_train)
    #     nn_label = loo.predict([X_test])
    #     if nn_label == y_test:
    #         correct += 1
    # print("USCHAD Catch-22 (All Features) LOO Accuracy: {}".format(correct/lendataset))

    # BEST (FEATURE) OF CATCH22
    # global_best_acc = 0
    # global_best_feature = featurenames[0]
    # for f in tqdm(range(22)):
    #     curr_acc = 0
    #     correct = 0
    #     data = np.expand_dims(featurevectors[:,f],axis=1)
    #     for test_index in tqdm(range(lendataset)):
    #         X_train, X_test = np.delete(data,test_index,axis=0), data[test_index]
    #         y_train, y_test = np.delete(labels,test_index), labels[test_index]
    #         loo.fit(X_train,y_train)
    #         nn_label = loo.predict([X_test])
    #         if nn_label == y_test:
    #             correct += 1
    #     curr_acc = correct/lendataset
    #     if curr_acc > global_best_acc:
    #         global_best_acc = curr_acc
    #         global_best_feature = featurenames[f]
    # print("USCHAD Catch-22 (Best of Features) LOO Accuracy, Feature: {}, {}".format(global_best_acc, global_best_feature))

    # FORWARD SELECTION ON CATCH22 FEATURES
    global_best_acc = 0
    use_features = []
    lastfeature = None
    featuresbank = np.arange(22)
    featurenamesbank = featurenames
    featurenames = np.array(featurenames)
    
    while(len(featuresbank)>0):
        # print("96: {}".format(use_features))
        global_best_feature_idx = None
        global_best_feature_name = ""
        for f in tqdm(range(len(featuresbank)), desc="Feature list run through:"):
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
    print("USCHAD Catch-22 (FS on Features) LOO Accuracy, Features: {}, {}".format(global_best_acc, np.array(featurenames[use_features])))

# all features: 0.44126984126984126
# best feature: USCHAD Catch-22 (Best of Features) LOO Accuracy, Feature: 0.36666666666666664, SP_Summaries_welch_rect_centroid
# USCHAD Catch-22 (FS on Features) LOO Accuracy, Features: 0.6492063492063492, ['SP_Summaries_welch_rect_centroid' 'PD_PeriodicityWang_th0_01'
#  'CO_f1ecac' 'CO_trev_1_num' 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1'
#  'IN_AutoMutualInfoStats_40_gaussian_fmmi'
#  'FC_LocalSimple_mean1_tauresrat' 'MD_hrv_classic_pnn40'
#  'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1' 'SB_MotifThree_quantile_hh'
#  'SP_Summaries_welch_rect_area_5_1']