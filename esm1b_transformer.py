#import torch
#model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
import multiprocessing
import random
from typing import Sequence

import esm
import numpy as np
import torch
from Bio import SeqIO
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from torchinfo import summary

from utils import esm1b_preprocessing, knn, read_fasta

#sample_seq = [
#    ("protein1", "AGEPKLDAGV"),
#    #("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#]
#def chunks(l, n):
#    """Yield successive n-sized chunks from l."""
#    for i in range(0, len(l), n):
#        yield l[i:i + n]

#seq_batches=chunks(sample_seq,5)
#sequence_representations = []

# for idx, suquence in enumerate(seq_batches):
#     print(idx)
#     sample_batch_labels, sample_batch_strs, sample_batch_tokens = batch_converter(sequence)

#     with torch.no_grad():
#         results = model(sample_batch_tokens, repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]#.detach().cpu()
    
#     for i, (_, seq) in enumerate(batch): 
#         sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

#     del token_representations, sample_batch_labels, sample_batch_strs, sample_batch_tokens, results
#     torch.cuda.empty_cache()

class esm1b():
    def __init__(self,seqFile):
        # Reach Sequences
        sequences = list(SeqIO.parse(seqFile, "fasta"))

        # Sequences to Dict fro ESM1b
        self.sample_seq = [] 
        for seq in sequences:
            s = str(seq.seq)
            if len(s) <= 1022: # Maximum allowed sequence length
                self.sample_seq.append((seq.id,s))
        print(len(self.sample_seq))

        # Load ESM-1b model
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        #model = model.to("cuda:1")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.sequence_representations = {}#np.empty(len(self.sample_seq))

    def esm1b_predict(self,sample_idx):
        id, sequence = self.sample_seq[sample_idx]
        input = [(id,sequence)]
        sample_batch_labels, sample_batch_strs, sample_batch_tokens = self.batch_converter(input)

        with torch.no_grad():
            results = self.model(sample_batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]#.detach().cpu()
        
        #for i, (_, seq) in enumerate(batch): 
        i = 0
        seq_rep = token_representations[i, 1 : len(sequence) + 1].mean(0)
        self.sequence_representations[id] = seq_rep
        return {id, seq_rep}

    def esm1b_multi_predict(self):
        a_pool = multiprocessing.Pool()
        #return a_pool.map(self.esm1b_predict, range(len(self.sample_seq)))
        return a_pool.map(self.esm1b_predict, range(10))

        #return result


sequences = SeqIO.parse("sequences_labels.fasta", "fasta")#.to_dict()
#seq_dict = SeqIO.to_dict(SeqIO.parse("sequences_labels.fasta", "fasta"))
#X_train, X_test = train_test_split(record_dict, test_size=(1/3))
#x = [str(s.seq) for s in sequences]
#y = [s.description[s.description.find("[")+1:s.description.rfind("]")] for s in sequences]
embedded_x_train = [] 
embedded_x_test = [] 
y_train = [] 
y_test = [] 

for s in sequences:
    if random.random() < 0.7: # Train
        embedded_x_train.append(torch.load("my_reprs/"+str(s.description)+".pt")['mean_representations'][33])
        y_train.append(s.description[s.description.find("[")+1:s.description.rfind("]")] )
    else: # Test
        embedded_x_test.append(torch.load("my_reprs/"+str(s.description)+".pt")['mean_representations'][33])
        y_test.append(s.description[s.description.find("[")+1:s.description.rfind("]")] )

svd = TruncatedSVD(n_components=100)
embedded_x_train = svd.fit_transform(torch.stack(embedded_x_train))
embedded_x_test = svd.transform(torch.stack(embedded_x_test))
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(1/3))
#embedded_x_train = [] 
knearestneighbors = knn(embedded_x_train, np.array(y_train,dtype=float))
knearestneighbors.multi_score(embedded_x_test, np.array(y_test,dtype=float))

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
