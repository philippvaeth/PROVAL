## https://github.com/amyxlu/CPCProt

import os
import os.path as path
import sys
import time

import torch

import CPCProt

CPCProt.__path__ = ['/home/vaethp/dev/ProtSeqEmb/CPCProt','/home/vaethp/dev/ProtSeqEmb/CPCProt/CPCProt']
from torchinfo import summary

from CPCProt.model.cpcprot import CPCProtModel
from CPCProt.model.heads import CPCProtEmbedding
from CPCProt.tokenizer import Tokenizer

ckpt_path = "CPCProt/data/best.ckpt"  # Replace with actual path to CPCProt weights
model = CPCProtModel()
model.load_state_dict(torch.load(ckpt_path))
embedder = CPCProtEmbedding(model)
tokenizer = Tokenizer()

# Example primary sequence
#seq = "AGEPKLDAGV"
seq = "LITRSVSRPLRYAVDIIEDIAQGNLRRDVSVTGKDEVSRLLAAMSSQRERLSA"

# Tokenize and convert to torch tensor
t=time.time()
input = torch.tensor([tokenizer.encode(seq)])   # (1, L)

# Pad if sequence length < 11 with zeros (=padding token)
n = 11-len(input[0])
if n > 0:
  input = torch.cat((input[0],torch.zeros(n,dtype=int)),dim=0).unsqueeze(0)


summary(embedder,input_data=input,depth=5,verbose=1,col_names=["input_size","output_size","num_params"],method="zmean")
summary(embedder,input_data=input,depth=5,verbose=1,col_names=["input_size","output_size","num_params"],method="cmean")
summary(embedder,input_data=input,depth=5,verbose=1,col_names=["input_size","output_size","num_params"],method="cfinal")
# We note three ways to obtain pooled embeddings from CPCProt.
# z_mean and c_mean are the averages of non-padded tokens in z and c, respectively.
# In our paper, we find that z_mean is best for tasks where local effects
# are important (e.g. deep mutational scanning tasks)
# c_final is the final position of the context vector.
# We find that this is best for tasks where global information
# is important (e.g. remote homology tasks).
z_mean = embedder.get_z_mean(input)   # (1, 512)
print(z_mean)
c_mean = embedder.get_c_mean(input)   # (1, 512)
print(c_mean)
c_final = embedder.get_c_final(input)  # (1, 512)
print(c_final)
# $z$ is the output of the CPCProt encoder
z = embedder.get_z(input)  # (1, L // 11, 512)

# $c$ is the output of the CPCProt autoregressor
c = embedder.get_c(input)  # (1, L // 11, 512)
print(time.time()-t)
