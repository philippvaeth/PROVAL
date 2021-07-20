#from tsnecuda import TSNE
from sklearn.manifold import TSNE
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

method = "pca" # [tsne, pca]

for filename in os.listdir("vecs"):
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
        reduced_vecs = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(vecs)
        print(reduced_vecs.shape)
    elif method == "pca":
        reduced_vecs = PCA(n_components=2).fit_transform(vecs)
        print(reduced_vecs.shape)
    else: raise ValueError

   

    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title=filename[:-2] )

    # Create the scatter
    ax.scatter(
        x=reduced_vecs[:,0],
        y=reduced_vecs[:,1],
        c=y,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.7,
        s=0.5)
    
    plt.savefig('{}/{}.jpg'.format(method,filename[:-2]))
    #plt.show()
    #fig.show()
    #break
