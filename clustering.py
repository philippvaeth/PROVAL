#from tsnecuda import TSNE
from sklearn.manifold import TSNE
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

method = "tsne" # [tsne, pca]

for filename in os.listdir("vecs"): #['sw_complex.p']
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/{}".format(filename),"rb"))
    #y = np.concatenate((np.ndarray(list(y_train.values())),np.adaray(list(y_test.values()))),0)
    y = np.concatenate((np.array(list(y_test.values())), np.array(list(y_train.values()))),0)

    #vecs = torch.cat((torch.tensor(list(embedded_x_test.values())).squeeze(1),torch.tensor(list(embedded_x_train.values())).squeeze(1)),0)
    # if list(embedded_x_test.values())[0].shape[0] == 1:
    #     vecs = np.concatenate((np.array(list(embedded_x_test.values())).squeeze(1),np.array(list(embedded_x_train.values())).squeeze(1)),0)
    # else:
    #     #list(np.array(x) for x in embedded_x_test.values())
    #     vecs = np.concatenate((np.array(list(np.squeeze(np.array(x)) for x in embedded_x_test.values())),np.array(list(np.squeeze(np.array(x)) for x in embedded_x_train.values()))),0)

        #vecs = np.concatenate((np.array(list(embedded_x_test.values())),np.array(list(embedded_x_train.values()))),0)

    vecs = np.concatenate((np.array(list(np.squeeze(np.array(x)) for x in embedded_x_test.values())),np.array(list(np.squeeze(np.array(x)) for x in embedded_x_train.values()))),0)

    #x = np.unique(vecs,axis=0)
    if method == "tsne":
        #reduced_vecs = TSNE(n_components=2, perplexity=35, learning_rate=200, n_iter=1000, verbose=1).fit_transform(vecs)
        reduced_vecs = TSNE(n_components=2).fit_transform(vecs)

        print(reduced_vecs.shape)
    elif method == "pca":
        reduced_vecs = PCA(n_components=2).fit_transform(vecs)
        print(reduced_vecs.shape)
    else: raise ValueError

   

    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    #ax = fig.add_subplot(1, 1, 1, title=filename[:-2] )
    ax = fig.add_subplot(1, 1, 1 )
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # Create the scatter
    ax.scatter(
        x=reduced_vecs[:,0],
        y=reduced_vecs[:,1],
        c=y,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.9,
        s=1)#0.5)
    
    plt.savefig('{}/{}.jpg'.format(method,filename[:-2]), transparent=True,dpi=400,)
    #plt.show()
    #fig.show()
    #break
