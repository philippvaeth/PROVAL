import numpy as np
import re
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import collections

def setup():
  os.system("pip install gdown")
  os.system("gdown 'https://drive.google.com/uc?id=1RHVICFAPIHjsb9pTI_X1mimDZBLSwdW2'")
  os.system("unzip 'uniprot.tab.zip'")
  os.system("git clone https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library.git")
  os.system("cd Complete-Striped-Smith-Waterman-Library/src;make")
  os.system("pip install biopython")
  
  
  def loadtomatrix(file, n, mode="BLAST"):
  sim_matrix = np.zeros((n,n),dtype=np.uint16)
  if mode == "SAM":
    pattern = "AS:i:(.*?)[ \t]"
  elif mode == "BLAST":
    pattern = "optimal_alignment_score: (.*?)s"
  
  with open(file) as my_file:
      for idx,line in enumerate(my_file):
        if mode == "BLAST" and idx%4 == 2:
          sw_score = re.search(pattern, line).group(1)
          #print(str(idx)+" ,"+str(idx//(n*4))+" ,"+str(idx//4%n))
          sim_matrix[idx//(n*4),idx//4%n] = int(sw_score)
        elif mode == "SAM":
          sw_score = re.search(pattern, line).group(1)
          sim_matrix[idx//n,idx%n] = int(sw_score)
  return sim_matrix

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)



#import numpy as np
class UniProt(Dataset):
  def __init__(self, file_name):
    self.file = pd.read_table(file_name).dropna().to_numpy()

    self.id = self.file[:,0]
    self.x = self.file[:,1]

    p = re.compile('([0-9]{7})')
    self.y = [np.array(re.findall(p,s),dtype=np.uint32) for s in self.file[:,2]]

    #assert(self.x.shape == self.y.shape)

data = UniProt("uniprot.tab")

def swm(input_file, output_file):
  os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format(input_file,input_file,output_file))

def getTopNSingleClassIds(data,top_classes):
    # Get IDs of single class sequences
    ids = []
    for idx,y in enumerate(data.y):
      if len(data.y[idx]) == 1:
        ids.append(idx)

    counts = collections.Counter(np.hstack(np.array(data.y)[ids]))
    # Get IDs for most common n classes with only one label
    most_common_classes = [item[0] for item in counts.most_common()[:top_classes]]
    #print(most_common_classes)
    ids_dict = {}
    for c in most_common_classes:
      ids = []
      for idx,y in enumerate(data.y):
        if len(data.y[idx]) == 1 and (data.y[idx]==[c]).all():
          ids.append(idx)
      #print(len(ids))
      ids_dict[c]=ids
    return ids_dict

def dataIdsToFastaFile(ids,output_file):
  records = [] 
  for n in ids:
    seq = Seq(data.x[n])
    record = SeqRecord(seq,id=data.id[n])
    records.append(record)

  SeqIO.write(records, output_file, "fasta")
  
  
input_file = "example.fasta"
output_file = "example.txt"
examples_per_class = 1000
top_classes = 15

prep_time = time.time()
ids_dict = getTopNSingleClassIds(data,top_classes)
ids = np.hstack([i[:examples_per_class] for i in ids_dict.values()])
if len(ids) != examples_per_class*top_classes:
  raise ValueError

dataIdsToFastaFile(ids,input_file)
print("PREPARATION--- %s seconds ---" % (time.time() - prep_time))

overallstart_time = time.time()
swm(input_file,output_file)
sim_matrix = loadtomatrix(output_file,examples_per_class*top_classes)
print("OVERALL--- %s seconds ---" % (time.time() - overallstart_time))
print(sim_matrix)
