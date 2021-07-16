# Source: https://github.com/peter-volkov/biovec/blob/master/protein_vectors_building/get_protein_vectors.py
#from . import protvec
#from . import protveckyu as protvec
#import biovec
import gensim

from utils import *


#from protveckyu import prot_vec, utils
def get_ngram_vectors(file_path):
    ngram_vectors = {}
    vector_length = None
    with open(file_path) as infile:
        for line in infile:
            line_parts = line.rstrip().split()   
            # skip first line with metadata in word2vec text file format
            if len(line_parts) > 2:     
                ngram, vector_values = line_parts[0], line_parts[1:]          
                ngram_vectors[ngram] = np.array(vector_values, dtype=np.float32)
    return ngram_vectors

def normalize(x):
    return x / np.sqrt(np.dot(x, x))


def get_protein_vector(protein_string, ngram_vectors, ngram_length=3):
    vector_length = 100
    ngrams_sum = np.zeros(vector_length, dtype=np.float32)
    for index in range(len(protein_string) + 1 - ngram_length):
        ngram = protein_string[index:index + ngram_length]
        if ngram in ngram_vectors:
            ngram_vector = ngram_vectors[ngram]
            ngrams_sum += ngram_vector
        else:
            print(ngram)
    return normalize(ngrams_sum)

ngram_vectors = get_ngram_vectors("temp/biovec/ngram_vector_building/uniprot_sprot.3gram_vectors.100d.txt")

#pv = biovec.models.load_protvec('biovec/trained_models/swissprot-reviewed-protvec.model')
# model = gensim.models.Word2Vec(
#         None, 
#         min_count=2, 
#         size=100, 
#         sg=int(True), 
#         window=25)

# model.wv.load_word2vec_format("temp/biovec/ngram_vector_building/uniprot_sprot.3gram_vectors.100d.txt")

#pv.to_vecs("AGAMQSASM")

train = read_fasta("train.fasta")
test = read_fasta("test.fasta")

embedded_x_train = [] 
embedded_x_test = [] 
y_train = [] 
y_test = [] 

for s in test:
    embedded_x_test.append(get_protein_vector(str(s.seq),ngram_vectors))
    y_test.append(s.description)


for s in train:
    embedded_x_train.append(get_protein_vector(str(s.seq),ngram_vectors))
    y_train.append(s.description)



knearestneighbors = knn(embedded_x_train, np.array(y_train,dtype=float))
knearestneighbors.multi_score(embedded_x_test, np.array(y_test,dtype=float))
# 0.7306
