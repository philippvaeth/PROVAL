import pandas as pd
import os
import re
import multiprocessing
from utils import read_fasta
from Bio import SeqIO
import subprocess
import time


class sw_multiprocessing(object):

    def __init__(self, train_fasta, test_fasta):
        super().__init__()
        self.query = read_fasta(test_fasta)
        self.database = read_fasta(train_fasta)
        self.train_fasta = train_fasta
        self.test_fasta = test_fasta
        self.pattern_target = "target_name: (.*?)\n"
        self.pattern_query = "query_name: (.*?)\n"
        self.pattern_score = "optimal_alignment_score: (.*?)s"

    def score_step(self, query_seq):
        output_file = 'temp/{}.txt'.format(query_seq.id)
        fasta_file = 'temp/{}.fasta'.format(query_seq.id)
        SeqIO.write([query_seq], fasta_file, "fasta")
        subprocess.run(
            'Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}'
            .format(fasta_file, self.train_fasta, output_file),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        df_row = pd.DataFrame(index=[query_seq.id],
                              columns=[seq.id for seq in self.database])
        with open(output_file) as my_file:
            for idx, line in enumerate(my_file):
                if idx % 4 == 0:  # Line 0: target
                    target = re.search(self.pattern_target, line).group(1)
                elif idx % 4 == 1:  # Line 1: query
                    query = re.search(self.pattern_query, line).group(1)
                elif idx % 4 == 2:  # Line 3: alignment score
                    sw_score = re.search(self.pattern_score, line).group(1)
                    df_row.at[target, query] = int(sw_score)

        os.remove(output_file)
        os.remove(fasta_file)
        return df_row

    def multi_score(self, query):
        a_pool = multiprocessing.Pool(30)
        df_rows = a_pool.map(self.score_step, query)
        self.sw_alignment_matrix = pd.concat(df_rows)


def alignSmithWaterman(mode="full"):
    if mode == "traintest":
        # Train (10000 x 10000)
        t = time.time()
        sw = sw_multiprocessing("data/train.fasta", "data/train.fasta")
        query = read_fasta("data/train.fasta")
        sw.multi_score(query)
        sw.sw_alignment_matrix.to_pickle("data/sw_alignment_train.pkl")
        assert sw.sw_alignment_matrix.isnull().values.any() == False
        print("Smith-Waterman TrainxTrain time:", time.time() - t)

        # Test x Train (5000 x 10000)
        t = time.time()
        sw = sw_multiprocessing("data/train.fasta", "data/test.fasta")
        query = read_fasta("data/test.fasta")
        sw.multi_score(query)
        assert sw.sw_alignment_matrix.isnull().values.any() == False
        sw.sw_alignment_matrix.to_pickle("data/sw_alignment_test.pkl")
        print("Smith-Waterman TestxTrain time:", time.time() - t)

    elif mode == "full":
        # Full Train+Test x Train+Test (15000 x 15000)
        all_sequences_fasta = "all_sequences.fasta"
        sequences = read_fasta("data/train.fasta") + read_fasta(
            "data/test.fasta")
        for seq in sequences:
            seq.description = "[{}]".format(seq.description)
        SeqIO.write(sequences, all_sequences_fasta, "fasta")
        t = time.time()
        sw = sw_multiprocessing(all_sequences_fasta, all_sequences_fasta)
        sw.multi_score(sequences)
        assert sw.sw_alignment_matrix.isnull().values.any() == False
        sw.sw_alignment_matrix.to_pickle("data/sw_alignment_all.pkl")
        os.remove(all_sequences_fasta)
        print("Smith-Waterman full matrix time:", time.time() - t)
