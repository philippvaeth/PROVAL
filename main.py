import pandas as pd

from utils import *

# sequences = SeqIO.parse("data/sequences_labels.fasta", "fasta")


# train, test = train_test_split(list(sequences), test_size=1/3)
# print(len(train))
# print(len(test))
# SeqIO.write(train, "train.fasta", "fasta")
# SeqIO.write(test, "test.fasta", "fasta")

# output_file = "test.txt"
# os.system("Complete-Striped-Smith-Waterman-Library/src/ssw_test -p {} {} > {}".format("train.fasta","test.fasta",output_file))

train = SeqIO.parse("train.fasta", "fasta")
test = SeqIO.parse("test.fasta", "fasta")

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

sw_alignment_matrix.to_pickle("sw_alignment_test.pkl")
print(sw_alignment_matrix)
