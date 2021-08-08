#from tsnecuda import TSNE
from sklearn.manifold import TSNE
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from utils import is_symmetric
import pandas as pd

method = "eigenspectrum" 

for filename in ['bilstm_vecs_proj.p']:#os.listdir("vecs"): 
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/{}".format(filename),"rb"))
    #y = np.concatenate((np.array(list(y_test.values())), np.array(list(y_train.values()))),0)
    vecs = np.concatenate((np.array(list(np.squeeze(np.array(x)) for x in embedded_x_test.values())),np.array(list(np.squeeze(np.array(x)) for x in embedded_x_train.values()))),0)

    euclidean_inner_product_matrix = vecs @ vecs.T #euclidean_distances(vecs)
    #assert is_symmetric(euclidean_inner_product_matrix) == True
    eigenvalues = np.linalg.eigvalsh(euclidean_inner_product_matrix)
    # if method == "tsne":
    #     #reduced_vecs = TSNE(n_components=2, perplexity=35, learning_rate=200, n_iter=1000, verbose=1).fit_transform(vecs)
    #     reduced_vecs = TSNE(n_components=2).fit_transform(vecs)

    #     print(reduced_vecs.shape)
    # elif method == "pca":
    #     reduced_vecs = PCA(n_components=2).fit_transform(vecs)
    #     print(reduced_vecs.shape)
    # else: raise ValueError

   

    # Create the figure
    fig, ax = plt.subplots()
    #fig = plt.figure( figsize=(8,8) )
    #ax = fig.add_subplot(1, 1, 1, title=filename[:-2] )
    #ax = fig.add_subplot(1, 1, 1 )
    # fig.gca().xaxis.set_major_locator(plt.NullLocator())
    # fig.gca().yaxis.set_major_locator(plt.NullLocator())
    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # # Create the scatter
    # ax.scatter(
    #     x=reduced_vecs[:,0],
    #     y=reduced_vecs[:,1],
    #     c=y,
    #     cmap=plt.cm.get_cmap('Paired'),
    #     alpha=0.9,
    #     s=1)#0.5)
    #eigenvalues = [2^id for id in range(15000)]

    #ax.bar(np.array(list(range(15000,0,-1))), abs(eigenvalues))
    #ax.axis([0, 15000, min(eigenvalues), max(eigenvalues)])
    #ax.autoscale(True)
    
    plt.yscale("log")
    #plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'

    eigenvalues = eigenvalues[-100:]/np.max(eigenvalues[-100:])
    # thresholding
    eigenvalues[eigenvalues < 1e-6] = 1e-6
    ax.set_ylim(1e-06,1e+1)
    
    ax.vlines(x=list(range(0,100)), ymin=0, ymax=eigenvalues, color='#007ACC', alpha=0.2, linewidth=2)
    ax.plot(list(range(0,100)), eigenvalues, "o", markersize=4, color='#007ACC', alpha=0.6)
    #ax.bar(list(range(0,100)),eigenvalues[-100:]/np.max(eigenvalues[-100:]), width=0.8)

    plt.savefig('{}/{}.jpg'.format(method,filename[:-2]), transparent=True,dpi=400,)
    #print(eigenvalues[-100:])
    #df = pd.DataFrame({'lab':list(range(15000,0,-1)), 'val':eigenvalues})
    #ax = df.plot.bar(x='lab', y='val', rot=0)
    #plt.savefig('eigenspectrum/test.pdf')

    #plt.show()
    #fig.show()
    #break
