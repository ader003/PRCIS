import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from architecturefns import *
from preprocessingandrecords import *

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_ac_subroutine(model, fref, label_colors, affinity, linkagetype, methodname, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=(12,4))
    dendrogram(linkage_matrix, **kwargs)
    ax = plt.gca()
    ax.set_title("Method: {}, Affinity: {}, Linkage: {}".format(methodname,affinity,linkagetype))
    ax.tick_params(axis='y', which='major')
    ax.tick_params(axis='x', colors='white', labelsize=8)
    
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])
        
    
def plotty_ac(savename, methodname, affinity, linkagetype, X, fref, label_colors):
    def llf(l):
        return "{}".format(fref[l])
    c2 = AgglomerativeClustering(n_clusters=None, distance_threshold=0, affinity=affinity, linkage=linkagetype) # np.mean(X)
    c2 = c2.fit(X)
    plot_ac_subroutine(c2, fref, label_colors, affinity, linkagetype, methodname, orientation='top', leaf_label_func=llf)
    if savename != None:
        plt.savefig(savename, format="svg")
    plt.show()


def plotty_dict(ts, d, d_idxs, spliceindices, savename):
    n_element = len(d)
    fig = plt.figure(tight_layout=True, figsize=(30,15))
    gs = gridspec.GridSpec(2, n_element)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(range(len(ts)), ts, alpha=0.7)
    ax.set_xlabel(savename)

    for l in spliceindices:
        ax.axvline(x=l)
    
    c = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'springgreen', 'crimson', 'indigo', 'royalblue', 'mediumvioletred', 'yellowgreen', 'chocolate', 'gold']
    for j in range(min(n_element, len(c))):
        d_elem = d[j]
        ax1 = fig.add_subplot(gs[1, j])
        ax1.plot(d_elem, color=c[j])
        idxs = d_idxs[j]
        ax.plot(range(idxs[0], idxs[1]), d_elem, color=c[j])
        if j == 0:
            for tick in ax.get_xticklabels():
                tick.set_rotation(55)
        fig.align_labels()
        
    plt.title("{} patterns total; maximum 15 shown".format(len(d_idxs)), fontsize=20)

    # plt.savefig(savename + '.svg', format="svg")
    # plt.savefig(savename + '.png', format="png")
    plt.close()
    