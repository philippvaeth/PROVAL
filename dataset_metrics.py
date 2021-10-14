from utils import read_fasta
import numpy as np
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from biovec.utils import split_ngrams

# Load data
train = read_fasta("data/train.fasta")
test = read_fasta("data/test.fasta")
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
amino_acid_counts = OrderedDict(sorted(Counter(amino_acids).items()))
print("Amino acids in the dataset: ",len(amino_acids))
print("Amino acids counts: ",amino_acid_counts)
fig, ax = plt.subplots()
ax.bar((amino_acid_counts.keys()),amino_acid_counts.values())
plt.savefig('{}/{}.jpg'.format("dataset_metrics","amino_acid_counts"), transparent=True,dpi=400,)

# 3-gram stats + histogram
n_gram_list1,n_gram_list2,n_gram_list3  = split_ngrams(amino_acids,3)
n_gram_list = n_gram_list1 + n_gram_list2 + n_gram_list3
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
ax.bar((n_gram_most_common.keys()),n_gram_most_common.values())
plt.savefig('{}/{}.jpg'.format("dataset_metrics","3gram_most_common"), transparent=True,dpi=400,)

# Class overview
#labels = ['oxidoreductase activity, id:16491', 'transmembrane transporter activity, id:22857', 'DNA binding, id:3677', 'RNA binding, id:3723', 'structural constituent of ribosome, id:3735', 'translation elongation factor activity, id:3746', 'peptidyl-prolyl cis-trans isomerase activity, id:3755', 'metal ion binding, id:46872',
#       'proton-transporting ATP synthase activity, rotational mechanism, id:46933', 'serine-type endopeptidase inhibitor activity, id:4867', 'structural molecule activity, id:5198', 'calcium ion binding, id:5509', 'ATP binding, id:5524', 'NADH dehydrogenase (ubiquinone) activity, id:8137', 'toxin activity, id:90729']
labels = ['GO:0016491', 'GO:0022857', 'GO:0003677', 'GO:0003723', 'GO:0003735', 'GO:0003746', 'GO:0003755', 'GO:0046872',
       'GO:0046933', 'GO:0004867', 'GO:0005198', 'GO:0005509', 'GO:0005524', 'GO:0008137', 'GO:0090729']
sizes = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
colors = {3677.0:'#e6194B', 3723.0:'#3cb44b', 3735.0:'#ffe119', 3746.0:'#4363d8', 3755.0:'#f58231', 4867.0:'#42d4f4', 5198.0:'#f032e6', 5509.0:'#fabed4', 5524.0:'#469990', 8137.0:'#dcbeff', 16491.0:'#9A6324', 22857.0:'#fffac8',46872.0:'#800000', 46933.0:'#aaffc3', 90729.0:'#000075'}

fig, ax = plt.subplots()
ax.pie(sizes, autopct='%1.1f%%',explode=explode,labels=labels,colors=colors.values(),
         startangle=90)
ax.axis('equal')
plt.savefig('{}/{}.jpg'.format("dataset_metrics","classes"), transparent=True,dpi=400,)

