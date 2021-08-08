import pickle
from numpy.lib.financial import nper
import os.path

import torch
from sklearn.decomposition import TruncatedSVD

from CPCProt.tokenizer import Tokenizer
from CPCProt import CPCProtModel, CPCProtEmbedding
from utils import knn_dict, read_fasta

import numpy as np

vec_type = 'zmean' # ['zmean','cmean','cfinal']

if os.path.isfile("vecs/cpcprot_vecs_{}_tsvd.p".format(vec_type)): 
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/cpcprot_vecs_{}_tsvd.p".format(vec_type),"rb"))
else:
    ckpt_path = "CPCProt/data/best.ckpt"  # Replace with actual path to CPCProt weights
    model = CPCProtModel()
    model.load_state_dict(torch.load(ckpt_path))
    embedder = CPCProtEmbedding(model)
    tokenizer = Tokenizer()

    # Example primary sequence
    train = read_fasta("train.fasta")
    test = read_fasta("test.fasta")

    embedded_x_train, embedded_x_test, y_train, y_test = {}, {}, {}, {} 

    def pad_sequence(seq, token=0):
        s = list(seq)
        while len(s) < 11:
            s.append(token)
        return np.array(s)

    for s in test:
        tokens = tokenizer.encode(s.seq)
        if len(tokens) < 11:
            tokens = pad_sequence(tokens)
        input = torch.tensor([tokens]) 

        if vec_type == 'zmean':
            vec = embedder.get_z_mean(input)   # (1, 512)
        elif vec_type== 'cmean':
            vec = embedder.get_c_mean(input)   # (1, 512)
        elif vec_type== 'cfinal':
            vec = embedder.get_c_final(input)  # (1, 512)
        else: raise ValueError

        embedded_x_test[str(s.id)] = vec.detach().cpu().squeeze(0)
        y_test[str(s.id)] = float(s.description)

    for s in train:
        tokens = tokenizer.encode(s.seq)
        if len(tokens) < 11:
            tokens = pad_sequence(tokens)
        input = torch.tensor([tokens]) 

        if vec_type == 'zmean':
            vec = embedder.get_z_mean(input)   # (1, 512)
        elif vec_type== 'cmean':
            vec = embedder.get_c_mean(input)   # (1, 512)
        elif vec_type== 'cfinal':
            vec = embedder.get_c_final(input)  # (1, 512)
        else: raise ValueError
        
        embedded_x_train[str(s.id)] = vec.detach().cpu().squeeze(0)
        y_train[str(s.id)] = float(s.description)
    
    svd = TruncatedSVD(n_components=100)
    vec_stack = torch.stack(list(embedded_x_train.values()))
    svd.fit(vec_stack)

    def truncate(item):
        vec = svd.transform(item[1].reshape(1,-1))
        return (item[0], vec)

    embedded_x_train = dict(map(truncate, embedded_x_train.items()))
    embedded_x_test = dict(map(truncate, embedded_x_test.items()))

    pickle.dump( [embedded_x_train, y_train, embedded_x_test, y_test], open( "vecs/cpcprot_vecs_{}_tsvd.p".format(vec_type), "wb" ) )



knearestneighbors = knn_dict(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)
# zmean tsvd: 0.777, 0.777
# cmean tsvd: 0.8332
# cfinal tsvd: 0.8222, 0.8222 , 0.8222