#from . import protvec
#from . import protveckyu as protvec
import biovec
from utils import *

#from protveckyu import prot_vec, utils

pv = biovec.models.load_protvec('biovec/trained_models/swissprot-reviewed-protvec.model')
#pv.to_vecs("AGAMQSASM")

train = read_fasta("train.fasta")
test = read_fasta("test.fasta")

embedded_x_train = {}#[] 
embedded_x_test = {}#[] 
y_train = {}#[] 
y_test = {}#[] 

for s in test:
    #embedded_x_test.append(pv.to_vecs(str(s.seq)))
    #y_test.append(s.description)
    embedded_x_test[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
    y_test[str(s.id)] = float(s.description)

for s in train:
    #embedded_x_train.append(pv.to_vecs(str(s.seq)))
    #y_train.append(s.description)
    embedded_x_train[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
    y_train[str(s.id)] = float(s.description)



#knearestneighbors = knn(embedded_x_train, np.array(y_train,dtype=float))
#knearestneighbors.multi_score(embedded_x_test, np.array(y_test,dtype=float))

knearestneighbors = knn_dict(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)
# 0.8132
