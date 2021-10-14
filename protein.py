import collections
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data import UniProt
from utils import *
#from sklearn.model_selection import train_test_split


#import numpy as np



input_file = "example.fasta"
output_file = "example.txt"
train_file = "data/train.fasta"
train_output_file = "train.txt"
test_file = "data/test.fasta"
test_output_file = "test.txt"
examples_per_class = 1000
top_classes = 15
setup = False

if setup:
  data = UniProt("uniprot.tab")
  prep_time = time.time()
  ids_dict = getTopNSingleClassIds(data,top_classes)
  ids = np.hstack([i[:examples_per_class] for i in ids_dict.values()])
  if len(ids) != examples_per_class*top_classes:
    raise ValueError

  train_ids, test_ids = train_test_split(ids, test_size=1/3, shuffle=True)
  dataIdsToFastaFile(train_ids,data,train_file)
  dataIdsToFastaFile(test_ids,data,test_file)
  print("PREPARATION--- %s seconds ---" % (time.time() - prep_time))

overallstart_time = time.time()

if train:
  train_sim_matrix = get_sw_similarity_matrix(train_file,train_file,train_output_file,examples_per_class*top_classes)
  train = torch.from_numpy(np.array(train_sim_matrix,dtype=np.float32))
  torch.save(train,"train.pt")
  test_sim_matrix = get_sw_similarity_matrix(test_file,train_file,test_output_file,examples_per_class*top_classes)
  test = torch.from_numpy(np.array(test_sim_matrix,dtype=np.float32))
  torch.save(test,"test.pt")
  print("OVERALL--- %s seconds ---" % (time.time() - overallstart_time))
  print(train.shape)
  print(test.shape)
  print(sim_matrix)

sim_matrix = torch.load("sim_matrix.pt")
train_sim, test_sim = train_test_split(sim_matrix,1/3)
print(train_sim.shape)
print(test_sim.shape)
torch.save(train_sim,"train_sim.pt")
torch.save(test_sim,"test_sim.pt")

#matrix = get_sw_similarity_matrix("seq1.fasta","seq2.fasta",1)
#print(matrix)