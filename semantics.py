from utils import knn
import os
import pickle

for filename in os.listdir("vecs"):
    embedded_x_train_dict, y_train, embedded_x_test_dict, y_test = pickle.load(open("vecs/{}".format(filename),"rb"))
    # Calculate accuracy scores
    knearestneighbors = knn(embedded_x_train_dict, y_train) # Save train values for comparison
    acc = knearestneighbors.multi_score(embedded_x_test_dict, y_test) # Calculate the distance from the test samles to all train samples + winner-takes-all rule for classification
    print("Embedding",filename[:-2],"classification accuracy",acc,"%")

# Results as reported:
# Embedding sw_complex classification accuracy 0.866 %
# Embedding protvec classification accuracy 0.8118 %
# Embedding cpcprot_cfinal_tsvd classification accuracy 0.8186 %
# Embedding sw_dissimrep classification accuracy 0.8478 %
# Embedding bilstm_vecs_proj classification accuracy 0.7828 %
# Embedding cpcprot_zmean_tsvd classification accuracy 0.7782 %
# Embedding esm1b_tsvd classification accuracy 0.9118 %