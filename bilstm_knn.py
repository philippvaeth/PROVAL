import pickle

from utils import *

embedded_x_train, y_train, embedded_x_test, y_test = pickle.load(open("vecs/bilstm_vecs_proj.p","rb"))

knearestneighbors = knn_dict(embedded_x_train, y_train)
knearestneighbors.multi_score(embedded_x_test, y_test)
# projector (100D): 0.7828, 0.7828, 0.7828 (mean 78.28, std 0)
# all hidden states (3705 D): 0.846
# all hidden states (3705 D) sum + truncated svd (100D): 0.836
# all hidden states (3705 D) avg + truncated svd (100D): 0.8236
