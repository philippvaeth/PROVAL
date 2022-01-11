from sklearn.manifold import TSNE
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

colors = {3677.0:'#e6194B', 3723.0:'#3cb44b', 3735.0:'#ffe119', 3746.0:'#4363d8', 3755.0:'#f58231', 4867.0:'#42d4f4', 5198.0:'#f032e6', 5509.0:'#fabed4', 5524.0:'#469990', 8137.0:'#dcbeff', 16491.0:'#9A6324', 22857.0:'#fffac8',46872.0:'#800000', 46933.0:'#aaffc3', 90729.0:'#000075'}

for filename in [_ for _ in os.listdir("vecs") if _ != 'sw_complex.p']: # tsne implementation does not support complex valued embeddings
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/{}".format(filename),"rb"))
    y = np.concatenate((np.array(list(y_test.values())), np.array(list(y_train.values()))),0)
    vecs = np.concatenate((np.array(list(np.squeeze(np.array(x)) for x in embedded_x_test.values())),np.array(list(np.squeeze(np.array(x)) for x in embedded_x_train.values()))),0)
    reduced_vecs = TSNE(n_components=2).fit_transform(vecs)
   
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1 )
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ycolors = [colors[c] for c in y]
    
    # Create the scatter
    scatter = ax.scatter(
        x=reduced_vecs[:,0],
        y=reduced_vecs[:,1],
        c=ycolors,
        alpha=0.8,
        s=1,)

    plt.savefig('tsne/{}.jpg'.format(filename[:-2]), transparent=True,dpi=400,)
