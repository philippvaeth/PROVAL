import pandas as pd

#from utils import *

#df = pd.read_pickle("sw_alignment_test.pkl").astype('int32')
import pickle

#from torch._C import dtype, float32
from utils import EmbedDataPelskaDuinApprox, sim2dis,EmbedDataComplexValued, read_fasta, knn_dict, EigenvalueCorrection, PelskaDuin
import numpy as np
from sklearn.decomposition import TruncatedSVD

train = read_fasta("train.fasta")
test = read_fasta("test.fasta")

sw_alignment_train = pickle.load(open("sw_alignment_train_train.pkl","rb"))
assert sw_alignment_train.isnull().values.any() == False
assert list(sw_alignment_train) == [_.id for _ in train]
assert list(sw_alignment_train.index) == [_.id for _ in train]
sw_alignment_train = sw_alignment_train.to_numpy(dtype=np.float)
sw_alignment_train = 0.5 * (sw_alignment_train + sw_alignment_train.T)
#sw_alignment_train_dis = sim2dis(sw_alignment_train)
y_train = {str(s.id):float(s.description) for s in train}

sw_alignment_test = pickle.load(open("sw_alignment_test.pkl","rb"))
assert sw_alignment_test.isnull().values.any() == False
assert list(sw_alignment_test) == [_.id for _ in train]
assert list(sw_alignment_test.index) == [_.id for _ in test]
sw_alignment_test = sw_alignment_test.to_numpy(dtype=np.float)
#sw_alignment_test_dis = sim2dis(sw_alignment_test.to_numpy(dtype=np.float))
y_test = {str(s.id):float(s.description) for s in test}

embedding_type = "pelskaduin" #["pelskaduinapprox", "complex", "eigcorrection", "tsvd"]
if embedding_type == "complex":
    embedded_x_train, embedded_x_test = EmbedDataComplexValued(sw_alignment_train,sw_alignment_test,100)
elif embedding_type == "pelskaduinapprox":
    embedded_x_train, embedded_x_test = EmbedDataPelskaDuinApprox(sw_alignment_train,sw_alignment_test,100)
elif embedding_type == "eigcorrection":
    embedded_x_train, embedded_x_test = EigenvalueCorrection(sw_alignment_train,sw_alignment_test,100)
elif embedding_type == "tsvd":
    tsvd = TruncatedSVD(n_components=100)
    embedded_x_train = tsvd.fit_transform(sw_alignment_train)
    embedded_x_test = tsvd.transform(sw_alignment_test)
elif embedding_type == "pelskaduin":
    embedded_x_train, embedded_x_test = PelskaDuin(sw_alignment_train,sw_alignment_test, 100)
    
embedded_x_train_dict = {str(s.id):vec for s,vec in zip(train,embedded_x_train)}
embedded_x_test_dict = {str(s.id):vec for s,vec in zip(test,embedded_x_test)}

pickle.dump( [embedded_x_train_dict, y_train, embedded_x_test_dict, y_test], open( "vecs/sw_{}.p".format(embedding_type), "wb" ) )


knearestneighbors = knn_dict(embedded_x_train_dict, y_train)
knearestneighbors.multi_score(embedded_x_test_dict, y_test)
# complex valued embedding (100D): 0.8674, 0.862, 0.8636 (mean 86.43, std 0.226)
# pelska duin embedding with random subsampling (100D): 0.872, 0.8654, 0.8626 (mean 86.66, std 0.394)
# eigenvalue correction + tsvd (100D): 0.7864, 0.7864, 0.7864 (mean 78.64, std 0)
# tsvd (100D): 0.855, 0.8546, 0.8548 (mean 85.48, std 0.016)
# pelska duin eimbedding with 100 largest eigenvalues + vecs (100D): 0.8504, 0.8504, 0.8504