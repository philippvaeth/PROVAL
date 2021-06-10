import collections
import os
import re
import sys

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def downloadGDriveFile(fileID, fileName):
  os.system("gdown 'https://drive.google.com/uc?id={}'".format(fileID))
  os.system("unzip '{}'".format(fileName))

def unzip(filename):
  pzf = PyZipFile(filename)
  pzf.extractall()

def setup():
  #os.system("pip install gdown")
  os.system("gdown 'https://drive.google.com/uc?id=1RHVICFAPIHjsb9pTI_X1mimDZBLSwdW2'")
  #os.system("unzip 'uniprot.tab.zip'")
  unzip("uniprot.tab.zip")
  os.system("git clone https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library.git")
  os.system("cd Complete-Striped-Smith-Waterman-Library/src;make")
  os.system("git clone https://github.com/tbepler/protein-sequence-embedding-iclr2019.git")
  os.system("cd protein-sequence-embedding-iclr2019/;python setup.py build_ext --inplace")
  os.system("wget http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz;tar -xf pretrained_models.tar.gz")
  #os.system("pip install biopython")
  #downloadGDriveFile("1uk128kGh1FDAjnV78gsM4UEvLjEnW13i", "sim_matrix.npy.zip")
  #os.system("unzip 'sim_matrix.npy.zip'")
  
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

def dataIdsToFastaFile(ids,data,output_file):
  records = [] 
  for n in ids:
    seq = Seq(data.x[n])
    record = SeqRecord(seq,id=data.id[n])
    records.append(record)

  SeqIO.write(records, output_file, "fasta")
 
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


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    #return np.allclose(a, a.T, rtol=rtol, atol=atol)
    return bool(torch.all(a.T == a))

def is_psd(matrix):
    return bool(torch.all(torch.eig(matrix)[0][:,0]>=0))

def swm(input_file,input_file2, output_file):
  os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format(input_file,input_file2,output_file))

def get_sw_similarity_matrix(input_file,input_file2,output_file,n,mode="BLAST"):
    swm(input_file,input_file2,output_file)
    return loadtomatrix(output_file,n, mode)

def truncatedSimilarityMatrix_SVD(matrix, dim):
  u,s,v = torch.svd(matrix.float().cuda())
  mask = torch.cat((torch.ones(dim),torch.zeros(s.shape()-dim))).cuda()
  u[:,:1000]*s[:1000]*v.T[:,:1000]
  return u*(s*mask)*v.T

def train_test_split_sim(sim_matrix,y, test_size):
  train_idx = int(len(sim_matrix[0])*(1-test_size))
  X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y, float(test_size))
  #test_idx = len(sim_matrix[0])*(split)
  print(train_idx)
  return X_train[:,:train_idx],X_test[:,:train_idx], y_train, y_test 

def make_psd(matrix):
  assert(is_symmetric(matrix)==True)
  u,s,v = torch.svd(matrix)
  if bool(torch.all(s>=0)):
    return torch.mm(u,s,v.T)
  else:
    s_corr = torch.sub(s,torch.min(s))
    return torch.mm(u,s_corr,v.T)

def read_fasta(fastaFile):
  return list(SeqIO.parse("sequences.fasta", "fasta"))

def esm1b_preprocessing(sequences: SeqIO):
  c=0
  for i in sequences: 
    if(len(i.seq)>1024):
      c+=1
  print(c)
  return [ (str(sequence.id),str(sequence.seq)) for sequence in sequences]

def dis2sim(dis):
    """
    :param dis: np.ndarray
    :return:
    """
    (n, m) = dis.shape
    if n != m:
        raise Exception('The dissimilarity matrix must be square.')
    if np.diag(dis).any():
        raise Exception('The dissimilarity matrix must have zero diagonal.')
    if not is_symmetric(dis):
        raise Exception('The dissimilarity matrix must be symmetric.')

    """
    Matlab Code:
    [N,N_]=size(Dis);
    J = eye(N) - repmat(1/N,N,N);
    Sim = -0.5 * J * Dis * J;
    Sim=(Sim+Sim')/2;
    """

    J = np.eye(n) - np.tile(1 / n, (n, n)) # np.tile: "Construct an array by repeating A the number of times given by reps."
    sim = -0.5 * J @ dis @ J
    sim = 0.5 * (sim + sim.T)

    return sim

def sim2dis(sim):
    (n, m) = sim.shape
    if n != m:
        raise Exception('The similarity matrix must be square.')
    if not is_symmetric(sim):
        raise Exception('The similarity matrix must be symmetric.')

    d1 = np.diag(sim)
    d2 = d1.T
    dis = (d1 + d2) - 2 * sim
    dis = 0.5 * (dis + dis.T)
    return dis
