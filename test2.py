
import torch
from sklearn import neighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch._C import device

from utils import *

s = torch.load("sim_matrix.pt")
# symmetrize
s = np.tril(s) + np.triu(s.T, 1)
# alignment -> dis
d = sim2dis(s)
# dis -> sim
#sim =  dis2sim(d)

y = np.zeros((15000))
for i in range(15000):
  y[i] = i // 1000

#train_sim, test_sim = train_test_split(sim_matrix,1/3)
X_train, X_test, y_train, y_test = train_test_split_sim(s, y, test_size=(1/3))

# Embed Complex Valued - 0.821
#embedded_x_train, embedded_x_test = EmbedDataComplexValued(X_train,X_test,100)

# Embed with truncated SVD - sim: 86,5 % val acc, dis: 79,86
#svd = TruncatedSVD(n_components=100)
#embedded_x_train = svd.fit_transform(X_train)
#embedded_x_test = svd.transform(X_test)

# Direct dissim matrix knn
#embedded_x_train = X_train
#embedded_x_test = X_test

# MDS
mds = MDS(n_components=100, dissimilarity="precomputed")
embedded_x_train = mds.fit_transform(X_train)
embedded_x_test = mds.transform(X_test)

knearestneighbors = knn(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)


#scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(embedded_x_train, y_train)
#score = knn.score(embedded_x_test,y_test)

# comparison
#print(sim)
#print(s)
#print(torch.all(torch.eq(torch.from_numpy(sim),torch.from_numpy(s))))

#s = torch.from_numpy(s).float().to("cuda:0")
#e_s, v_s  = torch.symeig(s,eigenvectors=True)
#print(e_s)
#print(torch.allclose(torch.matmul(v_s, torch.matmul(e_s.diag_embed(), v_s.t())), s))
#print(torch.eq(s,v_s@e_s.diag()@v_s.T))

#sim = torch.from_numpy(sim).float().to("cuda:0")
#e_sim, v_sim = torch.symeig(sim,eigenvectors=True)
#print(e_sim)
#print(torch.eq(sim,v_sim@e_sim.diag()@v_sim.T))

#eig_diff = e_s/e_sim
#print(eig_diff[eig_diff>1.5])



#w_s, v_s = np.linalg.eig(s)
#w_sim, v_sim = np.linalg.eig(sim)
#print(w_s)
#print(w_sim)

print("")
#print(d)
#print(is_symmetric(d))
#d = d*d.T
#print(d)
#print(is_symmetric(d))
#print(is_psd(d))
#d_psd = make_psd(d)

#d = torch.tril(d) + torch.triu(d.T, 1)
#print(torch.all(torch.eq(torch.from_numpy(d1),d)))
