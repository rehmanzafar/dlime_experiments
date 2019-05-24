import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.datasets import load_breast_cancer

from load_dataset import LoadDataset

bc_data = LoadDataset(which='bc')
ildp_data = LoadDataset(which='ildp')
hp_data = LoadDataset(which='hp')

plt.figure(figsize=(5, 4))
#plt.title("Dendograms")
clust = shc.linkage(bc_data.data.data, method='ward')
dend = shc.dendrogram(clust)
filename= 'results/dendrogram_bc.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 4))
#plt.title("Dendograms")
clust = shc.linkage(hp_data.data.data, method='ward')
dend = shc.dendrogram(clust)
filename= 'results/dendrogram_ildp.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 4))
#plt.title("Dendograms")
clust = shc.linkage(ildp_data.data.data, method='ward')
dend = shc.dendrogram(clust)
filename= 'results/dendrogram_hp.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()