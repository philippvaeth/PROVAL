import os, time, torch, pickle, argparse
from utils import read_fasta 
from embedding_utils import swTrainTest, swDissimRep, swDissimRepComplex, protvec, esm1b, cpcprot
from sklearn.decomposition import TruncatedSVD

parser = argparse.ArgumentParser(description='Embedding settings.')
parser.add_argument('--sw_alignment', type=str, default='full',
                    help='full: n+m x n+m matrix / traintest: nxn, mxn matrices')
#parser.add_argument('embedding_type', default="all",
#                    help='sum the integers (default: find the max)')
args = parser.parse_args()

train = read_fasta("train.fasta")
y_train = {str(s.id):float(s.description) for s in train}
test = read_fasta("test.fasta")
y_test = {str(s.id):float(s.description) for s in test}
d = 100
sw_alignment_train, sw_alignment_test = swTrainTest(train, test, args.sw_alignment)

for embedding_type in ["sw_dissimrep", "sw_complex", "protvec", "esm1b_tsvd", "cpcprot_cfinal_tsvd", "cpcprot_zmean_tsvd"]:
   if not os.path.isfile("vecs/{}.p".format(embedding_type)):  
        t = time.time()
        if embedding_type ==  "sw_dissimrep":
            embedded_X_Train, embedded_X_Test = swDissimRep(sw_alignment_train,sw_alignment_test, train, test, d)
        elif embedding_type ==  "sw_complex":
            embedded_X_Train, embedded_X_Test = swDissimRepComplex(sw_alignment_train,sw_alignment_test, train, test, d)
        elif embedding_type ==  "protvec":
            embedded_X_Train, embedded_X_Test = protvec(train, test)
        elif embedding_type ==  "esm1b_tsvd":
            embedded_X_Train, embedded_X_Test = esm1b(train, test)
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
                return (item[0], vec)

            embedded_X_Train = dict(map(truncate, embedded_X_Train.items()))
            embedded_X_Test = dict(map(truncate, embedded_X_Test.items()))

        print(embedding_type,"embedding time:",time.time()-t)

        # Save vectors for further processing
        pickle.dump( [embedded_X_Train, y_train, embedded_X_Test, y_test], open( "vecs/{}.p".format(embedding_type), "wb" ) )
