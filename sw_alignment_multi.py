import pandas as pd
import os
import re
import multiprocessing
from utils import read_fasta
from Bio import SeqIO

class sw_multiprocessing(object):
    def __init__(self, train_fasta, test_fasta):
        super().__init__()
        self.query = read_fasta(test_fasta)
        self.database = read_fasta(train_fasta)
        self.sw_alignment_matrix = pd.DataFrame(index=[seq.id for seq in self.query], columns=[seq.id for seq in self.database])
        self.train_fasta = train_fasta
        self.test_fasta = test_fasta
        self.pattern_target = "target_name: (.*?)\n"
        self.pattern_query = "query_name: (.*?)\n"
        self.pattern_score = "optimal_alignment_score: (.*?)s"
        
    def score_step(self,query_seq_id):
        output_file = "{}.txt".format(query_seq_id)
        fasta_file = "{}.fasta".format(query_seq_id)
        SeqIO.write(self.query[query_seq_id], fasta_file, "fasta")
        os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format(fasta_file,self.train_fasta,output_file))

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
                    self.sw_alignment_matrix.at[query,target] = int(sw_score)
        os.remove(output_file)
        os.remove(fasta_file)

    def multi_score(self, query):
        result = [self.score_step(seq.id) for seqid in query]
        #a_pool = multiprocessing.Pool()
        #a_pool.map(self.score_step, query.keys())



sw = sw_multiprocessing("train.fasta","test.fasta")
query = read_fasta("test.fasta")
sw.multi_score(query)

