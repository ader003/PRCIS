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

# def plotty_aae(savename,tses, dicts, idxs):
#     fig, axs = plt.subplots(2,figsize=(8,3))
#     colors = ['c','m']
#     for k in idxs.keys():
#         g = k[0]
#         pattern = dicts[g][0]
#         j = idxs[k][0]
#         x = np.arange(j[0],j[1])
#         y = np.asarray(pattern)
#         print(x,y)
#         axs[g].plot(np.arange(len(tses[g])),tses[g], label=k[1]+" TS", c='k')
#         axs[g].plot(x, y, label=k[1]+" Pattern", c=colors[g])
        
#     for g in range(2):
#         axs[g].grid()
#         handles, labels = axs[g].get_legend_handles_labels() # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
#         by_label = dict(zip(labels, handles))
#         axs[g].legend(by_label.values(), by_label.keys())
#     plt.savefig('/home/ader003/tsdict/paperfigs/pngs/'+savename+'.png', fmt='png')
#     plt.savefig('/home/ader003/tsdict/paperfigs/svgs/'+savename+'.svg', fmt='svg')


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


# def plot_scattercomp(savename, distmats, colorize): # last parameter is colors if you're using weallwalk
#     i = 0
#     fig, ax = plt.subplots(2,len(distmats), figsize=(62,20))
#     for k in distmats.keys():
#         # ic(k)
#         j = 0
#         distmat = distmats[k]

#         # MSD
#         mds = MDS(2, dissimilarity='precomputed')
#         mds.fit(distmat)
#         Y = mds.fit_transform(distmat)
#         x = Y[:,0]
#         y = Y[:,1]
#         ax[j,i].scatter(x,y, c=colorize)
#         ax[j,i].set_title(k+' MDS')
#         ax[j,i].grid()

#         j += 1
#         # TSNE
#         tsne = TSNE(n_components=2, metric='precomputed')
#         tsne.fit(distmat)
#         Y = tsne.fit_transform(distmat)
#         x = Y[:,0]
#         y = Y[:,1]
#         ax[j,i].scatter(x,y, c=colorize)
#         ax[j,i].set_title(k+' TSNE')
#         ax[j,i].grid()
#         i += 1
    
#     plt.savefig(savename + '.svg', format="svg")
#     plt.savefig(savename + '.png', format="png")
#     plt.close()


# def plotty_scatter(distmat, colors):
#     fig, ax = plt.subplots(1,2, figsize=(20,10))
#     plt.rcParams.update({'font.size': 16})
#     colorize = [colors[k] for k in colors.keys()]
#     labels = [k for k in colors.keys()]
#     # MSD
#     mds = MDS(2, dissimilarity='precomputed')
#     mds.fit(distmat)
#     Y = mds.fit_transform(distmat)
#     x = Y[:,0]
#     y = Y[:,1]
#     ax[0].scatter(x,y, c=colorize)
#     ax[0].set_title('MDS')
#     ax[0].grid()
#     for i, txt in enumerate(labels):
#         ax[0].annotate(txt[:-3], (x[i],y[i]))
#     # TSNE
#     tsne = TSNE(n_components=2, metric='precomputed')
#     tsne.fit(distmat)
#     Y = tsne.fit_transform(distmat)
#     x = Y[:,0]
#     y = Y[:,1]
#     ax[1].scatter(x,y, c=colorize)
#     ax[1].set_title('TSNE')
#     ax[1].grid()
#     for i, txt in enumerate(labels):
#         ax[1].annotate(txt[:-3], (x[i],y[i]))

#     plt.show()


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
    