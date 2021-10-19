import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import numpy as np

from load_dataset import LoadDataset

# bc_data = LoadDataset(which='bc')
ildp_data = LoadDataset(which='ildp')
# hp_data = LoadDataset(which='hp')
# thyroid_data = LoadDataset(which='thyroid')
#
# # plt.figure(figsize=(5, 4))
# # #plt.title("Dendograms")
# # clust = shc.linkage(thyroid_data.data.data, method='ward')
# # dend = shc.dendrogram(clust, no_labels = True) #, no_labels = True
# # plt.axhline(y=30, c='grey', lw=1, linestyle='dashed') # #723d46
# # filename= 'results/dendrogram_thyroid.pdf'
# # plt.savefig(filename, bbox_inches='tight')
# # plt.show()
# #
# plt.figure(figsize=(5, 4))
# #plt.title("Dendograms")
# clust = shc.linkage(bc_data.data.data, method='ward')
# dend = shc.dendrogram(clust, no_labels = True)
# plt.axhline(y=12500, c='grey', lw=1, linestyle='dashed') # #723d46
# filename= 'results/dendrogram_bc.pdf'
# plt.savefig(filename, bbox_inches='tight')
# plt.show()
# #
# plt.figure(figsize=(5, 4))
# #plt.title("Dendograms")
# clust = shc.linkage(hp_data.data.data, method='ward')
# dend = shc.dendrogram(clust, no_labels = True)
# plt.axhline(y=600, c='grey', lw=1, linestyle='dashed') # #723d46
# filename= 'results/dendrogram_hp.pdf'
# plt.savefig(filename, bbox_inches='tight')
# plt.show()
#
# plt.figure(figsize=(5, 4))
# #plt.title("Dendograms")
# clust = shc.linkage(ildp_data.data.data, method='ward')
# dend = shc.dendrogram(clust, no_labels = True)
# plt.axhline(y=42, c='grey', lw=1, linestyle='dashed') # #723d46
# # filename= 'results/dendrogram_ildp.pdf'
# # plt.savefig(filename, bbox_inches='tight')
# plt.show()

# ILDP dataset has outliers therefore, outliers were removed before
# plotting dendrogram

def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # print(IQR)
    df_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return np.array(df_out.iloc[:, 0:9])


X = remove_outliers(ildp_data.data.dataframe)

plt.figure(figsize=(5, 4))
# plt.title("Dendograms")
clust = shc.linkage(X, method='ward')
dend = shc.dendrogram(clust, no_labels=True)
plt.axhline(y=1100, c='grey', lw=1, linestyle='dashed')  # #723d46
# filename = 'results/dendrogram_ildp.pdf'
# plt.savefig(filename, bbox_inches='tight')
plt.show()
