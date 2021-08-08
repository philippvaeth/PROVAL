import pandas as pd
import os
import re
import multiprocessing
from utils import read_fasta
from Bio import SeqIO
import subprocess
import tqdm
import time

class sw_multiprocessing(object):
    def __init__(self, train_fasta, test_fasta):
        super().__init__()
        self.query = read_fasta(test_fasta)
        self.database = read_fasta(train_fasta)
        #self.sw_alignment_matrix = pd.DataFrame(index=[seq.id for seq in self.query], columns=[seq.id for seq in self.database])
        self.train_fasta = train_fasta
        self.test_fasta = test_fasta
        self.pattern_target = "target_name: (.*?)\n"
        self.pattern_query = "query_name: (.*?)\n"
        self.pattern_score = "optimal_alignment_score: (.*?)s"
        
    def score_step(self,query_seq):
        output_file = 'temp/{}.txt'.format(query_seq.id)
        fasta_file = 'temp/{}.fasta'.format(query_seq.id)
        SeqIO.write([query_seq], fasta_file, "fasta")
        subprocess.run('Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}'.format(fasta_file,self.train_fasta,output_file), shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        df_row = pd.DataFrame(index=[query_seq.id], columns=[seq.id for seq in self.database])
        with open(output_file) as my_file:
            for idx,line in enumerate(my_file):
                #print(idx,line)
                if idx%4 ==0: # Line 0: target
                    target = re.search(self.pattern_target, line).group(1)
                    #print(target)
                elif idx%4 ==1:  # Line 1: query
                    #print(line)
                    query = re.search(self.pattern_query, line).group(1)
                    #print(query)
                elif idx%4 == 2: # Line 3: alignment score
                    #print(line)
                    sw_score = re.search(self.pattern_score, line).group(1)
                    #print(sw_score)
                    #print(str(idx)+" ,"+str(idx//(n*4))+" ,"+str(idx//4%n))
                    #print(query,target,sw_score)
                    #self.sw_alignment_matrix.at[target,query] = int(sw_score)
                    df_row.at[target,query] = int(sw_score)

        os.remove(output_file)
        os.remove(fasta_file)
        return df_row
        #print(self.sw_alignment_matrix.iloc[:, 0].count()/len(self.sw_alignment_matrix.iloc[:, 0]))

    def multi_score(self, query):
        #result = [self.score_step(seq) for seq in query]
        a_pool = multiprocessing.Pool(30)
        df_rows = a_pool.map(self.score_step, query)
        self.sw_alignment_matrix = pd.concat(df_rows)
        self.sw_alignment_matrix.to_pickle("sw_alignment_traintest.pkl")

        #for _ in tqdm.tqdm(a_pool.imap_unordered(self.score_step, query), total=len(query)):
        #    pass

mode = "all" #['train', 'test', 'all']

if mode == "train":
    # Train (10000 x 10000) - 16 min
    sw = sw_multiprocessing("train.fasta","train.fasta")
    query = read_fasta("train.fasta")
    sw.multi_score(query)
    print(sw.sw_alignment_matrix)
    sw.sw_alignment_matrix.to_pickle("sw_alignment_train.pkl")
    print(sw.sw_alignment_matrix.isnull().values.any())
    assert sw.sw_alignment_matrix.isnull().values.any() == False

if mode == "test":
    # Test x Train (5000 x 10000)
    sw = sw_multiprocessing("train.fasta","test.fasta")
    query = read_fasta("test.fasta")
    sw.multi_score(query)
    assert sw.sw_alignment_matrix.isnull().values.any() == False
    print(sw.sw_alignment_matrix)
    sw.sw_alignment_matrix.to_pickle("sw_alignment_test.pkl")

if mode == "all":
    # Full Train+Test x Train+Test (15000 x 15000)
    t = time.time()
    all_sequences_fasta = "all_sequences.fasta"
    sequences = read_fasta("train.fasta") + read_fasta("test.fasta")
    for seq in sequences:
        seq.description = "[{}]".format(seq.description)
    SeqIO.write(sequences, all_sequences_fasta, "fasta")

    sw = sw_multiprocessing(all_sequences_fasta,all_sequences_fasta)
    print(len(sequences))
    sw.multi_score(sequences)
    assert sw.sw_alignment_matrix.isnull().values.any() == False
    print(sw.sw_alignment_matrix)
    sw.sw_alignment_matrix.to_pickle("sw_alignment_all.pkl")
    os.remove(all_sequences_fasta)
    print("Time:",time.time()-t)