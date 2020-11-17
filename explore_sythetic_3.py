from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from load_dataset import LoadDataset

test = LoadDataset(which='synth3')

X = test.data.data
y = test.data.target
feature_names = test.data.feature_names

train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)

#
# from sklearn.cluster import AgglomerativeClustering
# import numpy as np
#
# clustering = AgglomerativeClustering().fit(X)
# clustering
#
# clustering.labels_
#
# print(clustering.labels_)
#
# import matplotlib.pyplot as plt
# import scipy.cluster.hierarchy as shc
# plt.figure(figsize=(5, 4))
# #plt.title("Dendograms")
# clust = shc.linkage(test.data.data, method='ward')
# dend = shc.dendrogram(clust)
# # filename= 'results/dendrogram_hp.pdf'
# # plt.savefig(filename, bbox_inches='tight')
# plt.show()

clf = LogisticRegression(random_state=0)
clf.fit(train, labels_train)
#clf.predict(test, labels_test)

print(clf.score(test, labels_test))
#
# print(clf.coef_, clf.intercept_)