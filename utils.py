import collections
import multiprocessing
import os
import re
import sys

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split


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
    #return np.allclose(a, a.T, rtol=rtol, atol=atol
    a = torch.from_numpy(a)
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
  X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y, test_size=test_size)
  #test_idx = len(sim_matrix[0])*(split)
  #print(train_idx)
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
  seq_list = []
  sequences = SeqIO.parse(fastaFile, "fasta")
  for seq in sequences:
    seq.description = re.search("\[(.*?)\]",seq.description).group(1)
    seq_list.append(seq)
  return seq_list

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

def sim2dis_2(sim):
    (n, m) = sim.shape
    if n != m:
        raise Exception('The similarity matrix must be square.')
    if not is_symmetric(sim):
        raise Exception('The similarity matrix must be symmetric.')

    dis = np.diag(sim) - sim
    dis = (dis + dis.T)
    return dis

def EmbedDataComplexValued(X_Train, X_Test, SubsampleSizeM): 
    N = X_Train.shape[0]
    R = np.random.choice(N, SubsampleSizeM) # landmark selection can be done by k-means++ also for indefinite due to Oglic (2019)
    R.sort()
    X_Train_nm = X_Train[:,R]
    X_Train_mm = X_Train[np.ix_(R,R)] # symmetric subsample matrix from the original X_Train
    X_Test_nm = X_Test[:,R]
    
    # REMARK:
        # Here I simply skip Franks idea for large datasets and directly apply
        # an eigendecomposition of the subsample Matrix with the size m x m
        # This part can be kept uncommented for the first run. After the
        # technical report, we may include a nystrom-approximated
        # eigendecomposition in linear time...
    

    
    
                    #     if(EigendecompositionInLinearTime) # this works but is not needed - if W gets really huge we could use this or Halko
                    #         ny_K{1}=Knm(R,:);
                    #         ny_K{2} =W;
                    #         #[U,S,V] = svd(Knm,0);
                    #         [C, A]  = eig_ny(ny_K); # this is due to Gisbrecht(2015), but not
                    #         # needed here, providing a linear time eigendecomposition of a 
                    #         # nystroem-approximated kernel matrix (potentially non-psd)
                    #     
                    #     else   # the following line and the embedding is related to Landmark-MDS, due to Belongie/Fowlkes ECCV'2002 
                    #        [C, A]  = eig(Knm(R,:)); # this is the eigendecomposition of a subsample from the kernel matrix m x m, which in contrast to the ECCV-paper can be non-psd
                    #     end
                    

    
    # X_Train_IsFullRank = rank(X_Train) == N;
    
    if(False): # if input is Kmm - cheaper inverse is taken on a diagonal
        # we assume, that Kmm has full rank and hence has no zero eigenvalues 
        # M2=pinv(sqrt(A))*C'*Knm';
        
        
        # REMARK
        # Now we embed the 'left' eigenvectors into a (possibly complex
        # valued) vector space by taking the inverse of the
        # eigenvalue matrix
        # --> until now, everyone had to ensure that all eigenvalues have to be positve and hence took the absolute values of |diag(A)|.
        # However, this is associated with loss of information, because we
        # change the negative parts of the eigenspectrum.
        # We ignore the negative eigenvalues not and let the 'left'
        # eigenvalues become complex valued (implying that the complete
        # embedding matrix becomes complex valued) because we have your
        # great CGMLVW!
        
        # Here we have no need to calculate the inverse of X_Train_mm
        # explicitly as X_Train_mm is of full rank. Therefore, we can
        # simply determine the inverse on the diagonal of the eigenvalue
        # matrix.
        
        [C, A]  = eig(X_Train_mm)

        EmbeddedEigenvalues = diag(sqrt(1./diag(A)))
        
                #EmbeddedDataTrain=diag(sqrt(1./diag(A)))*C'*Knm'; # diag(sqrt(1./diag(A))) ~ is the right (or left, respectively) decomposition of Kmm
        
        # This step is moved outside of the if-else
                # EmbeddedDataTest = diag(sqrt(1./diag(A))) * C' * DataTest;
    else: # this is less cheap taking an inverse on Kmm to get W as seen in the above step
        
        
        # Matlab: W = pinv(X_Train_mm)
        W = np.linalg.pinv(X_Train_mm,hermitian=True)
        W = 0.5*(W+W.T); # making W symmetric - just in case if there has anything gone terribly wrong during the pinv()-operation
        # Matlab: [C,A] = eig(W)
        A,C = np.linalg.eigh(W)
        EmbeddedEigenvalues = np.diag(np.lib.scimath.sqrt(A))
    
    # Here we reconstruct the matrix but with the embedded eigenvalues.
    
    Embedded_X_Train = (EmbeddedEigenvalues @ C.T @ X_Train_nm.T).T
    
    # The Test data can easily embedded in the new vector space by using
    # the embedded eigenvalues and eigenvectors.
    Embedded_X_Test = (EmbeddedEigenvalues @ C.T @ X_Test_nm.T).T
    # Kapprox = X_Train*W*X_Train';
    return Embedded_X_Train, Embedded_X_Test


def mydist(x,y):
    return np.linalg.norm(x-y)

class knn(object):
  def __init__(self,x_train, y_train, neighbors=3):
      super().__init__()
      self.x_train = x_train 
      self.y_train = y_train
      self.neighbors = neighbors

  def score(self, x_test, y_test):
    acc = 0

    for idx,test_row in enumerate(x_test):
      distances = []
      for train_row in self.x_train:
        distances.append(np.linalg.norm(train_row-test_row))
      top_n = np.asarray(distances).argsort()[:self.neighbors]
      labels = self.y_train[top_n]
      #wta
      pred = np.bincount(labels.astype(int)).argmax()
      if pred == y_test[idx]: acc +=1
    return pred/y_test.shape[0]

  def score_step(self,test_row_idx):
    distances = []
    for train_row in self.x_train:
      distances.append(np.linalg.norm(train_row-self.x_test[test_row_idx]))
    top_n = np.asarray(distances).argsort()[:self.neighbors]
    labels = self.y_train[top_n]
    #wta
    pred = np.bincount(labels.astype(int)).argmax()
    #print(pred)
    #print(self.y_test[test_row_idx])
    return int(float(pred) == self.y_test[test_row_idx])

  def multi_score(self, x_test, y_test):
      self.y_test=y_test
      self.x_test=x_test
      a_pool = multiprocessing.Pool()
      result = a_pool.map(self.score_step, range(len(x_test)))
      print(np.sum(result)/self.y_test.shape[0])

class knn_dict(object):
  def __init__(self,x_train, y_train, neighbors=3):
      super().__init__()
      self.x_train = x_train 
      self.y_train = y_train
      self.neighbors = neighbors

  def score(self, x_test, y_test):
    acc = 0

    for idx,test_row in enumerate(x_test):
      distances = []
      for train_row in self.x_train:
        distances.append(np.linalg.norm(train_row-test_row))
      top_n = np.asarray(distances).argsort()[:self.neighbors]
      labels = self.y_train[top_n]
      #wta
      pred = np.bincount(labels.astype(int)).argmax()
      if pred == y_test[idx]: acc +=1
    return pred/y_test.shape[0]

  def score_step(self,test_row_idx):
    distances = {}#[]
    for train_sequence_id, train_sequence_vec in self.x_train.items():
      #distances.append(np.linalg.norm(train_row-self.x_test[test_row_idx]))
      distances[train_sequence_id] = np.linalg.norm(train_sequence_vec-self.x_test[test_row_idx])
    #top_n = np.asarray(distances).argsort()[:self.neighbors]
    top_n = sorted(distances, key=distances.get)[:self.neighbors]
    labels = [int(self.y_train[id]) for id in top_n]
    #wta
    pred = np.bincount(labels).argmax()
    #print(pred)
    #print(self.y_test[test_row_idx])
    return int(float(pred) == self.y_test[test_row_idx])

  def multi_score(self, x_test, y_test):
    self.y_test=y_test
    self.x_test=x_test
    result = [self.score_step(key) for key in x_test.keys()]
    #a_pool = multiprocessing.Pool()
    #result = a_pool.map(self.score_step, x_test.keys())
    print(np.sum(result)/len(self.y_test))
