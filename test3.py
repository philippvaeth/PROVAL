import numpy as np
import torch

from utils import sim2dis, sim2dis_2

# m = np.array([[16,2,1,3],
#               [2,6,1,11],
#               [1,1,12,1],
#               [3,11,1,4]])
# print(m)
# dis = sim2dis(m)
# print(dis)
# dis2 = sim2dis_2(m)
# print(dis2)
# dis3 = np.empty_like(m)
# for i in range(m.shape[0]):
#     for j in range(m.shape[1]):
#         dis3[i,j] = m[i,i] + m[j,j] - 2 * m[i,j]
# print(dis3)
s = torch.load("data/sim_matrix.pt")
# symmetrize
s = np.tril(s) + np.triu(s.T, 1)
#dis = sim2dis(s)
#print(dis)
dis2 = sim2dis_2(s)
print(dis2)

# dis3 = np.empty_like(s)
# for i in range(s.shape[0]):
#     for j in range(s.shape[1]):
#         dis3[i,j] = s[i,i] + s[j,j] - 2 * s[i,j]
# print(dis3)
# print(torch.allclose(torch.eq(torch.from_numpy(dis),torch.from_numpy(dis2))))
# print(torch.allclose(torch.eq(torch.from_numpy(dis),torch.from_numpy(dis3))))
