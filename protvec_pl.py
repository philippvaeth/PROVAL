#https://github.com/ChristophRaab/deep_transfer_learning/blob/17355065fca52be0848fa7d01514f39adf94ed0f/dda/sentiment/sentqs_preprocess_pytorch.py
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.trainer import data_loading
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

"""
 https://github.com/jowoojun/biovec/blob/2e3f86d744752eb89ae8c7ebe77d112c2efe5b17/word2vec/models.py
 'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
"""
def split_ngrams(seq, n):
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def generate_corpus(sequences_fname, n=3):
    tokenized_corpus = []
    for r in SeqIO.parse(sequences_fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        for ngram_pattern in ngram_patterns:
            tokenized_corpus.append(ngram_pattern)
    return tokenized_corpus


def seq2idxpair(seq,word2idxdict,window_size=25,n=3):
    ngram_seq = split_ngrams(seq, n)
    indices = [word2idx[ngram] for ngram in ngram_seq]
    idx_pairs = []
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))
    return idx_pairs

class SkipgramModel(LightningModule):

    def __init__(self, vocabulary_size, embedding_dim=100):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.inputlayer = nn.Linear(vocabulary_size, embedding_dim)  # equivalent to Dense in keras
        self.outputlayer = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, x):
        #print(x.shape)
        #x = F.softmax(self.inputlayer(x),dim=0)
        x = self.inputlayer(x)
        x = F.softmax(self.outputlayer(x),dim=0)
        #x = x.view(-1, 128)
        return x

    def training_step(self, batch, batch_idx):
        #print(batch)
        #print(batch[0].shape)
        #exit()
        #data, target = batch[0]
        data = batch[0][:,0]
        target = batch[0][:,1]
        #print(data)
        x = torch.zeros((len(data),self.vocabulary_size),dtype=torch.float32, device=self.device)
        for idx, d in enumerate(data):
            x[idx,d] = 1.0

        pred = self(x)

        y = torch.zeros((len(target),self.vocabulary_size),dtype=torch.float32, device=self.device)
        for idx, d in enumerate(target):
            y[idx,d] = 1.0

        #loss = nn.functional.binary_cross_entropy(pred, y)
        loss = nn.BCELoss()(pred, y)

        #loss = F.nll_loss(pred, y)
        #self.log(loss)
        #anchors_embeddings, target_embeddings, negative_embeddings = self(anchors, targets)
        #loss = negative_sampling_loss(anchors_embeddings, target_embeddings, negative_embeddings)
        #tensorboard_logs = {"training_loss": loss}
        return {"loss": loss}

    def configure_optimizers(self):
        return (
            torch.optim.Adam(self.parameters())
        )

if os.path.isfile("protvec/saved/word2idx.npy") and os.path.isfile("protvec/saved/idx_pairs.npy"):
    word2idx = np.load("protvec/saved/word2idx.npy",allow_pickle=True)
    idx_pairs = np.load("protvec/saved/idx_pairs.npy",allow_pickle=True)
    vocabulary_size = len(word2idx.item().keys())
else:
    # get corpus from fasta file
    tokenized_corpus = generate_corpus("sequences.fasta")
    print(len(tokenized_corpus))
    # build vocab
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    #idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)

    print(word2idx)
    # Make training data pairs
    window_size = 25
    idx_pairs = []
    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs) 

model = SkipgramModel(vocabulary_size)

ds = TensorDataset(torch.from_numpy(idx_pairs)) # create your datset
dl = DataLoader(ds,num_workers=4,batch_size=32) 

#summary(model,input_data=next(iter(dl))[0],depth=3,verbose=1,col_names=["input_size","output_size","num_params"])
summary(model,input_size=(32,8567),depth=3,verbose=1,col_names=["input_size","output_size","num_params"])

#trainer = Trainer(gpus=[0],min_epochs=10)
trainer = Trainer(min_epochs=10)

trainer.fit(model, dl)
#print(dataloader)
        # for epo in range(epochs):
        #     loss_val = 0
        #     for data,target in generate_data(corpus, window_size, V):
        #         data,target = torch.as_tensor(data).float().to(device), torch.as_tensor(target).float().to(device)
        #         optimizer.zero_grad()
        #         #x = get_input_layer(data).to(device)
        #         #y = get_input_layer(target).to(device)
        #         outputs = model(data)
        #         #loss = nn.functional.nll_loss(outputs, target)
        #         loss = nn.functional.binary_cross_entropy(outputs, target)
        #         loss_val += loss.item()
        #         loss.backward()
        #         optimizer.step()
