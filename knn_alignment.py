import pandas as pd

from utils import *

df = pd.read_pickle("sw_alignment_test.pkl").astype('int32')
train = read_fasta("train.fasta")
test = read_fasta("test.fasta")

if df.isnull().values.any() == True:
    raise Exception('The matrix must not contain nan values.')

#d = sim2dis_2(df)
#print(df)
#max = df.idxmax(axis=1) # find the sequence id with the maximum alignment score in the train sequences for each test sequence
#print(max)
#print(df.nlargest(1, df.columns))
#ef get_knn_label(sequence, k=3):
    #for _, row in df.iterrows():
 #       idx = sequence.nlargest(3)
        #print(idx)
  #      top_n_labels = [seq.description for seq in train if seq.id in idx.index]
        #print(top_n_labels)
   #     return np.bincount(np.array(top_n_labels).astype(int)).argmax()
#print(max)

k = 1 # neighbors for kNN
c = 0 # count the number of correct predictions

for test_idx, test_sequence_row in df.iterrows():
    y_test = int(next(seq.description for seq in test if seq.id == test_idx)) # get the description, i.e., the label, for the first test sequence that matches the test_sequence id
    
    train_idx = test_sequence_row.nlargest(k)
    top_n_labels = [seq.description for seq in train if seq.id in train_idx.index]
    y_train = np.bincount(np.array(top_n_labels).astype(int)).argmax() # wta
    #y_train = next(seq.description for seq in train if seq.id == max_train_sequence) # get the description, i.e., the label, for the first train sequence that matches the max_train_sequence id
    if y_test == y_train: c+=1

print("Accuracy: ",c/df.shape[0])
# d = df.copy(deep=True)

# for i in range(d.shape[0]):
#     for j in range(d.shape[1]):
#         d.iat[i,j] = df.iat[i,i] + df.iat[j,j] - 2 * df.iat[i,j]

# #print(d)
# d.to_pickle("dissimilarity.pkl")
# print(d)
