## Classification problem generation
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from load_dataset import LoadDataset

synth_data = LoadDataset(which='synth', n_sample=1000, n_features=20)
X = synth_data.data.data

#feature_names = test.data.feature_names
#target_names = test.data.target_names


train, test, labels_train, labels_test = train_test_split(synth_data.data.data, synth_data.data.target, train_size=0.80)

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(train, labels_train)

score = rf.score(test, labels_test)
print(f'Prediction Score: {score}')

clustering = AgglomerativeClustering().fit(X)
clustering.fit_predict(synth_data.data.data)

plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
plt.show()

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(5, 4))
#plt.title("Dendograms")
clust = shc.linkage(synth_data.data.data, method='ward')


dend = shc.dendrogram(clust)
filename= 'results/dendrogram_synthetic.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()
