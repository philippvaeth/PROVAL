import os, time, torch, pickle, argparse
from utils import read_fasta 
from embedding_utils import swTrainTest, swDissimRep, swDissimRepComplex, protvec, esm1b, cpcprot
from sklearn.decomposition import TruncatedSVD
import numpy as np

parser = argparse.ArgumentParser(description='Embedding settings.')
parser.add_argument('--sw_alignment', type=str, default='full',
                    help='full: n+m x n+m matrix / traintest: nxn, mxn matrices')
args = parser.parse_args()

train = read_fasta("data/train.fasta")
test = read_fasta("data/test.fasta")
d = 100
sw_alignment_train, sw_alignment_test = swTrainTest(train, test, args.sw_alignment)

for embedding_type in ["sw_dissimrep", "sw_complex", "protvec", "esm1b_tsvd", "cpcprot_cfinal_tsvd", "cpcprot_zmean_tsvd"]:
   if not os.path.isfile("vecs/{}.p".format(embedding_type)):  
        t = time.time()
        y_train = {str(s.id):float(s.description) for s in train}
        y_test = {str(s.id):float(s.description) for s in test}
        if embedding_type ==  "sw_dissimrep":
            embedded_X_Train, embedded_X_Test = swDissimRep(sw_alignment_train,sw_alignment_test, train, test, d)
        elif embedding_type ==  "sw_complex":
            embedded_X_Train, embedded_X_Test = swDissimRepComplex(sw_alignment_train,sw_alignment_test, train, test, d)
        elif embedding_type ==  "protvec":
            embedded_X_Train, y_train, embedded_X_Test, y_test = protvec(train, test)
        elif embedding_type ==  "esm1b_tsvd":
            embedded_X_Train, y_train, embedded_X_Test, y_test = esm1b(train, test)
        elif embedding_type ==  "cpcprot_cfinal_tsvd":
            embedded_X_Train, embedded_X_Test = cpcprot(train, test, "cfinal")
        elif embedding_type ==  "cpcprot_zmean_tsvd":
            embedded_X_Train, embedded_X_Test = cpcprot(train, test, "zmean")

        if len(list(embedded_X_Train.values())[0]) > d:
            svd = TruncatedSVD(n_components=d)
            vec_stack = torch.stack([_ for _ in embedded_X_Train.values()])
            svd.fit(vec_stack)

            def truncate(item):
                vec = svd.transform(item[1].reshape(1,-1))
                return (item[0], np.squeeze(vec))

            embedded_X_Train = dict(map(truncate, embedded_X_Train.items()))
            embedded_X_Test = dict(map(truncate, embedded_X_Test.items()))

        print(embedding_type,"embedding time:",time.time()-t)

        # Save vectors for further processing
        pickle.dump( [embedded_X_Train, y_train, embedded_X_Test, y_test], open( "vecs/{}.p".format(embedding_type), "wb" ) )
