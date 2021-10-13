#from tsnecuda import TSNE
from sklearn.manifold import TSNE
import os
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from sklearn.decomposition import PCA

for filename in os.listdir("vecs"):
    print(filename)
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
    reduced_vecs = TSNE(n_components=2).fit_transform(vecs)
   
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    #ax = fig.add_subplot(1, 1, 1, title=filename[:-2] )
    ax = fig.add_subplot(1, 1, 1 )
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    #colors = {3677.0:"#e1d9bc",3723.0:"#dda8ea",3735.0:"#a6b56b",3746.0:"#44b8e2",3755.0:"#fed992",4867.0:"#5ce5ff",5198.0:"#ffc69e",5509.0:"#46d8d3",5524.0:"#ffc3d1",8137.0:"#a1ffeb",16491.0:"#d29dad",22857.0:"#fffcb8",46872.0:"#86b1d1",46933.0:"#99b388",90729.0:"#77b995"}
    #colors = {3677.0:"#ff5578",3723.0:"#19bd5f",3735.0:"#340072",3746.0:"#b7e26d",3755.0:"#4179f2",4867.0:"#b28300",5198.0:"#eb73e0",5509.0:"#93e794",5524.0:"#142661",8137.0:"#dbd976",16491.0:"#ac97df",22857.0:"#486b00",46872.0:"#6b1b00",46933.0:"#02ddba",90729.0:"#826e00"}
    colors = {3677.0:'#e6194B', 3723.0:'#3cb44b', 3735.0:'#ffe119', 3746.0:'#4363d8', 3755.0:'#f58231', 4867.0:'#42d4f4', 5198.0:'#f032e6', 5509.0:'#fabed4', 5524.0:'#469990', 8137.0:'#dcbeff', 16491.0:'#9A6324', 22857.0:'#fffac8',46872.0:'#800000', 46933.0:'#aaffc3', 90729.0:'#000075'}
    ycolors = [colors[c] for c in y]
    #print(ycolors)
    # Create the scatter
    scatter = ax.scatter(
        x=reduced_vecs[:,0],
        y=reduced_vecs[:,1],
        c=ycolors,
        #cmap=plt.cm.get_cmap('Paired'),
        alpha=0.8,
        s=1,)#0.5)
    #legend_array = []    
    #price_change =  price_change ** 2 / (1 + price_change ** 2)

    #for label,color in colors.items():
    #    print(label,color)
    #    legend_array.append(mpatches.Patch(color=color, label=str(label)))
    #print(np.unique(np.array(y)).astype(int))
    #print(np.unique(np.array(y)).astype(str))
    #plt.legend(handles=legend_array)
    #plt.legend(handles=colors.values(),labels=colors.keys())
    #legend1 = ax.legend(np.unique(np.array(y)),
   #                 loc="upper left", title="Classes")
    #ax.add_artist(legend1)
    #ax.legend()
    #print(len(np.unique(np.array(y))))
    #plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(np.array(y)).all())

    #fig.legend(y)
    plt.savefig('tsne/{}.jpg'.format(filename[:-2]), transparent=True,dpi=400,)
    #plt.show()
    #fig.show()
    #break

from utils import read_fasta
import numpy as np
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from biovec.utils import split_ngrams

# Load data
train = read_fasta("train.fasta")
test = read_fasta("test.fasta")
data = train + test

# General sequence stats
raw_sequences = [str(seq.seq) for seq in data]
sequence_lenths = [len(s) for s in raw_sequences]
print("Average sequence length: ",sum(sequence_lenths)/len(raw_sequences))
print("Standard deviation sequence length: ",np.std(sequence_lenths))
print("Minimum sequence length: ",max(sequence_lenths))
print("Maximum sequence length: ",min(sequence_lenths))

# Amino acid stats
amino_acids = "".join(raw_sequences)
#amino_acid_counts = OrderedDict(Counter(amino_acids).most_common())
amino_acid_counts = OrderedDict(sorted(Counter(amino_acids).items()))
print("Amino acids in the dataset: ",len(amino_acids))
print("Amino acids counts: ",amino_acid_counts)
fig, ax = plt.subplots()
ax.bar((amino_acid_counts.keys()),amino_acid_counts.values()) #, width=0.8
plt.savefig('{}/{}.jpg'.format("dataset_metrics","amino_acid_counts"), transparent=True,dpi=400,)

# 3-gram stats + histogram
n_gram_list1,n_gram_list2,n_gram_list3  = split_ngrams(amino_acids,3)
n_gram_list = n_gram_list1 + n_gram_list2 + n_gram_list3
#n_gram_counts = OrderedDict(Counter(n_gram_list).most_common())
n_gram_counts = OrderedDict(sorted(Counter(n_gram_list).items()))
print("unique 3-grams in the dataset: ",len(np.unique(n_gram_list)))
print("3-grams occ <= 1000: ",len(np.array(list(n_gram_counts.values()))[np.array(list(n_gram_counts.values())) <= 1000])/len(np.array(list(n_gram_counts.values())))
)
print("Amino acids counts: ",n_gram_counts)
fig, ax = plt.subplots()
ax.bar((n_gram_counts.keys()),n_gram_counts.values()) 
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.savefig('{}/{}.jpg'.format("dataset_metrics","3gram_counts"), transparent=True,dpi=400,)

# most common 3-grams
n_gram_most_common =  OrderedDict(Counter(n_gram_list).most_common()[:15])
fig, ax = plt.subplots()
ax.bar((n_gram_most_common.keys()),n_gram_most_common.values()) #, width=0.8
plt.savefig('{}/{}.jpg'.format("dataset_metrics","3gram_most_common"), transparent=True,dpi=400,)

# Class overview
#labels = ['oxidoreductase activity, id:16491', 'transmembrane transporter activity, id:22857', 'DNA binding, id:3677', 'RNA binding, id:3723', 'structural constituent of ribosome, id:3735', 'translation elongation factor activity, id:3746', 'peptidyl-prolyl cis-trans isomerase activity, id:3755', 'metal ion binding, id:46872',
#       'proton-transporting ATP synthase activity, rotational mechanism, id:46933', 'serine-type endopeptidase inhibitor activity, id:4867', 'structural molecule activity, id:5198', 'calcium ion binding, id:5509', 'ATP binding, id:5524', 'NADH dehydrogenase (ubiquinone) activity, id:8137', 'toxin activity, id:90729']
labels = ['GO:0016491', 'GO:0022857', 'GO:0003677', 'GO:0003723', 'GO:0003735', 'GO:0003746', 'GO:0003755', 'GO:0046872',
       'GO:0046933', 'GO:0004867', 'GO:0005198', 'GO:0005509', 'GO:0005524', 'GO:0008137', 'GO:0090729']
sizes = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
fig, ax = plt.subplots()
ax.pie(sizes, autopct='%1.1f%%',explode=explode,labels=labels,#colors=colors,labels=labels,shadow=True,
         startangle=90,colors=y#,cmap=plt.cm.get_cmap('Paired')
         )
ax.axis('equal')
plt.savefig('{}/{}.jpg'.format("dataset_metrics","classes"), transparent=True,dpi=400,)

print(len(data))