# from sklearn.neighbors import NearestCentroid
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
#
# clf = NearestCentroid()
#
# clf.fit(X, y)
#
# print(clf.predict([[-0.8, -1]]))

import numpy as np
train = np.load("data/X_train_ildp.npy")
test = np.load("data/X_test_ildp.npy")
labels_train = np.load("data/y_train_ildp.npy")
labels_test = np.load("data/y_test_ildp.npy")

#samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors()
nbrs = neigh.fit(train)

distances, indices = nbrs.kneighbors(test)