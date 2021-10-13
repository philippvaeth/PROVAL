from os import times
import pandas as pd

from utils import *
import os.path
import pickle
import time

# sequences = SeqIO.parse("data/sequences_labels.fasta", "fasta")


# train, test = train_test_split(list(sequences), test_size=1/3)
# print(len(train))
# print(len(test))
# SeqIO.write(train, "train.fasta", "fasta")
# SeqIO.write(test, "test.fasta", "fasta")

# output_file = "test.txt"
# os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format("train.fasta","test.fasta",output_file))
train_fasta = "train.fasta"
train = SeqIO.parse(train_fasta, "fasta")
test_fasta = "test.fasta"
test = SeqIO.parse(test_fasta, "fasta")

# TRAIN
t = time.time()
output_file = "train.txt"
output_matrix_file = "sw_alignment_train.pkl"

if not os.path.isfile(output_file): 
  os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format(train_fasta,train_fasta,output_file))

if not os.path.isfile(output_matrix_file): 
  sw_alignment_matrix = pd.DataFrame(index=[seq.id for seq in train], columns=[seq.id for seq in train]) #  (10000 x 10000)

  pattern_target = "target_name: (.*?)\n"
  pattern_query = "query_name: (.*?)\n"
  pattern_score = "optimal_alignment_score: (.*?)s"
  file = "train.txt"

  with open(file) as my_file:
        for idx,line in enumerate(my_file):
          #print(idx,line)
          if idx%4 ==0: # Line 0: target
              target = re.search(pattern_target, line).group(1)
              #print(target)
          elif idx%4 ==1:  # Line 1: query
              #print(line)
              query = re.search(pattern_query, line).group(1)
              #print(query)
          elif idx%4 == 2: # Line 3: alignment score
            #print(line)
            sw_score = re.search(pattern_score, line).group(1)
            #print(sw_score)
            #print(str(idx)+" ,"+str(idx//(n*4))+" ,"+str(idx//4%n))
            #print(query,target,sw_score)
            sw_alignment_matrix.at[query,target] = int(sw_score)

  sw_alignment_matrix.to_pickle(output_matrix_file)
else:
  sw_alignment_matrix = pickle.load(open(output_matrix_file,"rb"))

print(sw_alignment_matrix)
print(time.time()-t)
exit
# TEST

output_file = "test.txt"
output_matrix_file = "sw_alignment_test.pkl"

if not os.path.isfile(output_file): 
  os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format(train_fasta,test_fasta,output_file))

if not os.path.isfile(output_matrix_file): 
  sw_alignment_matrix = pd.DataFrame(index=[seq.id for seq in test], columns=[seq.id for seq in train]) #  (5000 x 10000)

  pattern_target = "target_name: (.*?)\n"
  pattern_query = "query_name: (.*?)\n"
  pattern_score = "optimal_alignment_score: (.*?)s"
  file = "test.txt"

  with open(file) as my_file:
        for idx,line in enumerate(my_file):
          #print(idx,line)
          if idx%4 ==0: # Line 0: target
              target = re.search(pattern_target, line).group(1)
              #print(target)
          elif idx%4 ==1:  # Line 1: query
              #print(line)
              query = re.search(pattern_query, line).group(1)
              #print(query)
          elif idx%4 == 2: # Line 3: alignment score
            #print(line)
            sw_score = re.search(pattern_score, line).group(1)
            #print(sw_score)
            #print(str(idx)+" ,"+str(idx//(n*4))+" ,"+str(idx//4%n))
            print(query,target,sw_score)
            sw_alignment_matrix.at[query,target] = int(sw_score)

  sw_alignment_matrix.to_pickle(output_matrix_file)
else:
  sw_alignment_matrix = pickle.load(open(output_matrix_file,"rb"))

print(sw_alignment_matrix)
