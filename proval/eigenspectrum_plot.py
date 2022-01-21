import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

for filename in os.listdir("vecs"):
    if filename.endswith(".p"):
        embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(
            open("vecs/{}".format(filename), "rb"))
        vecs = np.concatenate(
            (np.array(
                list(np.squeeze(np.array(x)) for x in embedded_x_test.values())),
            np.array(
                list(np.squeeze(np.array(x))
                    for x in embedded_x_train.values()))), 0)
        euclidean_inner_product_matrix = vecs @ vecs.T
        eigenvalues = np.linalg.eigvalsh(euclidean_inner_product_matrix)

        # Create the figure
        fig, ax = plt.subplots()
        plt.yscale("log")
        plt.rcParams['axes.edgecolor'] = '#333F4B'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333F4B'
        plt.rcParams['ytick.color'] = '#333F4B'

        eigenvalues = eigenvalues[-100:] / np.max(eigenvalues[-100:])
        # thresholding
        eigenvalues[eigenvalues < 1e-6] = 1e-6
        ax.set_ylim(1e-06, 1e+1)
        ax.vlines(x=list(range(0, 100)),
                ymin=0,
                ymax=eigenvalues,
                color='#007ACC',
                alpha=0.2,
                linewidth=2)
        ax.plot(list(range(0, 100)),
                eigenvalues,
                "o",
                markersize=4,
                color='#007ACC',
                alpha=0.6)

        plt.savefig(
            'eigenspectrum/{}.jpg'.format(filename[:-2]),
            transparent=True,
            dpi=400,
        )
