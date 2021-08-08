#import torch
#model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
import multiprocessing
import os.path
import pickle
import random
from typing import Sequence

import esm
import numpy as np
import torch
from Bio import SeqIO
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from torchinfo import summary

from utils import esm1b_preprocessing, knn, knn_dict, read_fasta

if os.path.isfile("vecs/esm1b_vecs_tsvd.p") and False: 
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/esm1b_vecs_tsvd.p","rb"))
else:
    train = read_fasta("train.fasta")
    test = read_fasta("test.fasta")

    embedded_x_train, embedded_x_test, y_train, y_test = {}, {}, {}, {} 

    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to("cuda:1")
    batch_converter = alphabet.get_batch_converter()

    for s in test:
        if len(s.seq) <= 1022:
            _, _, sample_batch_tokens = batch_converter([(s.id,s.seq)])

            with torch.no_grad():
                results = model(sample_batch_tokens.to("cuda:1"), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]#.detach().cpu()
                
            seq_rep = token_representations[0, 1 : len(s.seq) + 1].mean(0)

            embedded_x_test[str(s.id)] = seq_rep.detach().cpu()
            y_test[str(s.id)] = float(s.description)

    for s in train:
        if len(s.seq) <= 1022:
            _, _, sample_batch_tokens = batch_converter([(s.id,s.seq)])

            with torch.no_grad():
                results = model(sample_batch_tokens.to("cuda:1"), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]#.detach().cpu()
                
            seq_rep = token_representations[0, 1 : len(s.seq) + 1].mean(0)

            embedded_x_train[str(s.id)] = seq_rep.detach().cpu()
            y_train[str(s.id)] = float(s.description)

    svd = TruncatedSVD(n_components=100)
    vec_stack = torch.stack([_ for _ in embedded_x_train.values()])
    svd.fit(vec_stack)

    def truncate(item):
        vec = svd.transform(item[1].reshape(1,-1))
        return (item[0], vec)

    embedded_x_train = dict(map(truncate, embedded_x_train.items()))
    embedded_x_test = dict(map(truncate, embedded_x_test.items()))

    pickle.dump( [embedded_x_train, y_train, embedded_x_test, y_test], open( "vecs/esm1b_vecs_tsvd.p", "wb" ) )

knearestneighbors = knn_dict(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)
# trunc (100D): 0.9328, 0.9328, 0.9328 (mean 93.28, std 0)
# full (1280D): 0.947



#knearestneighbors = knn(embedded_x_train, np.array(y_train,dtype=float))
#knearestneighbors.multi_score(embedded_x_test, np.array(y_test,dtype=float))

#summary(model,input_data=sample_batch_tokens,depth=2,verbose=1,col_names=["input_size","output_size","num_params"])
#batch_converter = alphabet.get_batch_converter()
#print(alphabet.tok_to_idx)
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)


#esm_model = esm1b("sequences_labels.fasta")
#esm_model.esm1b_predict(0)
#r = [esm_model.esm1b_predict(i) for i in range(5)]
#esm_model.esm1b_multi_predict()
#results = esm_model.sequence_representations
#print(results)
#torch.save("esm1b_1280d.pt",results)
#print(len(results))



#print(esm_model.sequence_representations[0].shape)
#sequenceList = read_fasta("sequences.fasta")
#data = esm1b_preprocessing(sequenceList)

#batch_labels, batch_strs, batch_tokens = batch_converter(data)
#print(batch_tokens.shape)
# model.cuda()
# sequence_representations = []
# for batch in batch_tokens:
#     with torch.no_grad():
#         results = model(batch.cuda(), repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]
#     print(token_representations.shape)
#     exit()
# Extract per-residue representations (on CPU)

#rint(token_representations.shape)
#exit()
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.


#print(len(sequence_representations))
#print(sequence_representations[0].shape)
