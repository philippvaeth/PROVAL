import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from utils import *
#from sklearn.model_selection import train_test_split
sim_matrix = torch.load("sim_matrix.pt")
sim_matrix = torch.stack([_/_.max() for _ in sim_matrix])
#error_ids = (sim != sim.T).nonzero()

#print(torch.sum([torch.abs(sim_matrix[x[0],x[1]]-sim_matrix[x[1],x[0]]) for x in error_ids]))

#abs_dif = torch.as_tensor([torch.abs(sim[x[0],x[1]]-sim[x[1],x[0]]) for x in ids])
#rel_dif = torch.as_tensor([(sim[x[0],x[1]]/sim[x[1],x[0]])-1 if sim[x[0],x[1]]>sim[x[1],x[0]] else sim[x[1],x[0]]/sim[x[0],x[1]]-1 for x in ids])

y = np.zeros((15000))
for i in range(15000):
  y[i] = i // 1000

#train_sim, test_sim = train_test_split(sim_matrix,1/3)
X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y, test_size=(1/3))
train_idx = int(len(sim_matrix[0])*(1-(1/3)))
X_train, X_test = X_train[:,:train_idx],X_test[:,:train_idx]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

svd = TruncatedSVD(n_components=100)
train_vec = svd.fit_transform(X_train)
print(train_vec.shape)
test_vec = svd.transform(X_test)
print(test_vec.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_vec, y_train)
#pred = knn.predict(test_vec)
print(knn.score(test_vec,y_test))
#print(np.sum((pred==y_test))/len(y_test))