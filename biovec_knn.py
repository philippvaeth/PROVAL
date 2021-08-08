import biovec
from utils import *
import pickle

if os.path.isfile("vecs/protvec_vecs.p") and False: 
    embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/protvec_vecs.p","rb"))
else:
    pv = biovec.models.load_protvec('biovec/trained_models/swissprot-reviewed-protvec.model')

    train = read_fasta("train.fasta")
    test = read_fasta("test.fasta")

    embedded_x_train, embedded_x_test, y_train, y_test = {},{},{},{}

    for s in test:
        embedded_x_test[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
        y_test[str(s.id)] = float(s.description)

    for s in train:
        embedded_x_train[str(s.id)] = sum(pv.to_vecs(str(s.seq)))
        y_train[str(s.id)] = float(s.description)

    pickle.dump( [embedded_x_train, y_train, embedded_x_test, y_test], open( "vecs/protvec_vecs.p", "wb" ) )


#knearestneighbors = knn(embedded_x_train, np.array(y_train,dtype=float))
#knearestneighbors.multi_score(embedded_x_test, np.array(y_test,dtype=float))

knearestneighbors = knn_dict(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)
# 0.8132, 0.8132,  0.8132 (mean  81.32, std 0)
