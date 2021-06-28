from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

class UniProt(Dataset):
  def __init__(self, file_name):
    self.file = pd.read_table(file_name).dropna().to_numpy()

    self.id = self.file[:,0]
    self.x = self.file[:,1]

    p = re.compile('([0-9]{7})')
    self.y = [np.array(re.findall(p,s),dtype=np.uint32) for s in self.file[:,2]]

    #assert(self.x.shape == self.y.shape)