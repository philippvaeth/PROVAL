import pandas as pd

from utils import *

df = pd.read_pickle("sw_alignment_test.pkl")

if df.isnull().values.any() == True:
    raise Exception('The matrix must not contain nan values.')

#d = sim2dis_2(df)
#print(d)

d = df.copy(deep=True)

for i in range(d.shape[0]):
    for j in range(d.shape[1]):
        d.iat[i,j] = df.iat[i,i] + df.iat[j,j] - 2 * df.iat[i,j]

#print(d)
d.to_pickle("dissimilarity.pkl")
print(d)
