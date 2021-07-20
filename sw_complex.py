import pandas as pd

from utils import *

df = pd.read_pickle("sw_alignment_test.pkl").astype('int32')
train = read_fasta("train.fasta")
test = read_fasta("test.fasta")
print(len(train))
print(len(test))
print(df.shape)
# sanity check
assert list(df) == [_.id for _ in train]
assert list(df.index) == [_.id for _ in test]

print(df)

# #print(train)

