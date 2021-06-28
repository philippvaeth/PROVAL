# src: https://github.com/jowoojun/biovec/blob/2e3f86d744752eb89ae8c7ebe77d112c2efe5b17/word2vec/models.py
import gzip
import os
import sys

import numpy as np
from Bio import SeqIO
from gensim.models import word2vec

"""
 'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
"""
def split_ngrams(seq, n):
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams


'''
Args:
    corpus_fname: corpus file name
    n: the number of chunks to split. In other words, "n" for "n-gram"
    out: output corpus file path
Description:
    Protvec uses word2vec inside, and it requires to load corpus file
    to generate corpus.
'''
def generate_corpusfile(corpus_fname, n, out):
    f = open(out, "w")
    #with gzip.open(corpus_fname, 'rb') as fasta_file:
    for r in SeqIO.parse(fasta_file, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
            sys.stdout.write(".")

    f.close()


def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)

def normalize(x):
    return x / np.sqrt(np.dot(x, x))

class ProtVec(word2vec.Word2Vec):

    """
    Either fname or corpus is required.

	corpus_fname: fasta file for corpus
    corpus: corpus object implemented by gensim
    n: n of n-gram
    out: corpus output file path
    min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
    """
    def __init__(self, corpus_fname=None, corpus=None, n=3, vector_size=100,
                 out="corpus.txt",  sg=1, window=25, min_count=1, workers=30):
        skip_gram = True

        self.n = n
        self.vector_size = vector_size
        self.corpus_fname = corpus_fname
        self.sg = int(skip_gram)
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.out = out

        directory = out.split('/')[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("directory(trained_models) created\n")

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if corpus_fname is not None:
            print ('Now we are checking whether corpus file exist')
            if not os.path.isfile(out):
                print ('INFORM : There is no corpus file. Generate Corpus file from fasta file...')
                generate_corpusfile(corpus_fname, n, out)
            else:
                print( "INFORM : File's Existence is confirmed")
            self.corpus = word2vec.Text8Corpus(out)
            print ("\n... OK\n")

    def word2vec_init(self, ngram_model_fname):
        word2vec.Word2Vec.__init__(self, self.corpus, vector_size=self.vector_size, sg=self.sg, window=self.window, min_count=self.min_count, workers=self.workers)
        model = word2vec.Word2Vec([line.rstrip().split() for line in open(self.out)], min_count = 1, vector_size=self.vector_size, sg=self.sg, window=self.window)
        model.wv.save_word2vec_format(ngram_model_fname)
    
    def to_vecs(self, seq, ngram_vectors):
        ngrams_seq = split_ngrams(seq, self.n)

        protvec = np.zeros(self.vector_size, dtype=np.float32)
        for index in range(len(seq) + 1 - self.n):
            ngram = seq[index:index + self.n]
            if ngram in ngram_vectors:
                ngram_vector = ngram_vectors[ngram]
                protvec += ngram_vector
        return normalize(protvec)
        
    def get_ngram_vectors(self, file_path):
        ngram_vectors = {}
        vector_length = None
        with open(file_path) as infile:
            for line in infile:
                line_parts = line.rstrip().split()   
                # skip first line with metadata in word2vec text file format
                if len(line_parts) > 2:     
                    ngram, vector_values = line_parts[0], line_parts[1:]          
                    ngram_vectors[ngram] = np.array(map(float, vector_values), dtype=np.float32)
        return ngram_vectors

fasta_file = "data/uniprot_sprot.fasta"
#Pfam_file = "document/Pfam-A.fasta.gz"
ngram_corpus_fname = "trained_models/ngram_vector.csv"
model_ngram = "trained_models/ngram_model"
sequences_file = "sequences.fasta"
#protein_vector_fname = "trained_models/protein_vector.csv"
#uniprot_with_families = "trained_models/uniprot_with_families.fasta"
#protein_pfam_vector_fname = "trained_models/protein_pfam_vector.csv"

#Make corpus
pv = ProtVec(fasta_file, out="trained_models/ngram_corpus.txt")

print ("Checking the file(trained_models/ngram_vector.csv)")
if not os.path.isfile(ngram_corpus_fname):
    print ('INFORM : There is no vector model file. Generate model files from data file...')
    
    #Make ngram_vector.txt and word2vec model
    pv.word2vec_init(ngram_corpus_fname)
    pv.save(model_ngram) 

    #Get ngram and vectors
    ngram_vectors = pv.get_ngram_vectors(ngram_corpus_fname)
    np.save("ngram_vectors.npy",ngram_vectors)
    
    protein_vecs = {}
    for record in SeqIO.parse(sequences_file, "fasta"):
        protein_name = record.name.split(' ')[-1]
        protein_vector = pv.to_vecs(record.seq, ngram_vectors)
        protein_vecs[protein_name] = protein_vector
    print(protein_vecs)
