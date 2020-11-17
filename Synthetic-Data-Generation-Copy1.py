## Classification problem generation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from itertools import combinations
from math import ceil

'''
data3 = make_classification(n_samples=20, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, 
                            n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, 
                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
'''

data3 = make_classification(n_samples=10000, n_features=20, n_informative=2, n_redundant=0, n_repeated=0,
                            n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=0)

df3 = pd.DataFrame(data3[0],columns=['x'+str(i) for i in range(1,21)])
df3['y'] = data3[1]
print(df3.head())




# lst_var=list(combinations(df3.columns[:-1],2))
# len_var = len(lst_var)
# len_var = 6
# plt.figure(figsize=(18,10))
# for i in range(1,len_var+1):
#     plt.subplot(2,ceil(len_var/2),i)
#     var1 = lst_var[i-1][0]
#     var2 = lst_var[i-1][1]
#     plt.scatter(df3[var1],df3[var2],s=200,c=df3['y'],edgecolor='k')
#     plt.xlabel(var1,fontsize=14)
#     plt.ylabel(var2,fontsize=14)
#     plt.grid(True)
#
# plt.show()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(data3[0])

target_ids = range(len(df3.columns))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g'
for i, c, label in zip(target_ids, colors, df3.columns):
    plt.scatter(X_2d[data3[1] == i, 0], X_2d[data3[1] == i, 1], c=c, label=label)
plt.legend()
plt.show()

#### Making class separation easy by tweaking `class_sep`


data3 = make_classification(n_samples=20, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                            n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=3.0, 
                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
df3 = pd.DataFrame(data3[0],columns=['x'+str(i) for i in range(1,5)])
df3['y'] = data3[1]

lst_var=list(combinations(df3.columns[:-1],2))
len_var = len(lst_var)
plt.figure(figsize=(18,10))
for i in range(1,len_var+1):
    plt.subplot(2,ceil(len_var/2),i)
    var1 = lst_var[i-1][0]
    var2 = lst_var[i-1][1]
    plt.scatter(df3[var1],df3[var2],s=200,c=df3['y'],edgecolor='k')
    plt.xlabel(var1,fontsize=14)
    plt.ylabel(var2,fontsize=14)
    plt.grid(True)


# ### Making class separation hard by tweaking `class_sep`

data3 = make_classification(n_samples=20, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                            n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=0.5, 
                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
df3 = pd.DataFrame(data3[0],columns=['x'+str(i) for i in range(1,5)])
df3['y'] = data3[1]

lst_var=list(combinations(df3.columns[:-1],2))
len_var = len(lst_var)
plt.figure(figsize=(18,10))
for i in range(1,len_var+1):
    plt.subplot(2,ceil(len_var/2),i)
    var1 = lst_var[i-1][0]
    var2 = lst_var[i-1][1]
    plt.scatter(df3[var1],df3[var2],s=200,c=df3['y'],edgecolor='k')
    plt.xlabel(var1,fontsize=14)
    plt.ylabel(var2,fontsize=14)
    plt.grid(True)


# ### Making data noisy by increasing `flip_y`

plt.figure(figsize=(18,10))
for i in range(6):
    data3 = make_classification(n_samples=20, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, 
                                n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.1*i, class_sep=1.0, 
                                hypercube=True, shift=0.0, scale=1.0, shuffle=False, random_state=101)
    df3 = pd.DataFrame(data3[0],columns=['x'+str(i) for i in range(1,5)])
    df3['y'] = data3[1]
    plt.subplot(2,3,i+1)
    plt.title(f"Plot for flip_y={round(0.1*i,2)}")
    plt.scatter(df3['x1'],df3['x2'],s=200,c=df3['y'],edgecolor='k')
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.grid(True)


# ### Plot datasets with varying degree of class separation

plt.figure(figsize=(18,5))
df2 = pd.DataFrame(data=np.zeros((20,1)))
for i in range(3):
    data2 = make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, 
                                n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0, class_sep=i+0.5, 
                                hypercube=True, shift=0.0, scale=1.0, shuffle=False, random_state=101)
    df2['x'+str(i+1)+'1']=data2[0][:,0]
    df2['x'+str(i+1)+'2']=data2[0][:,1]
    df2['y'+str(i+1)] = data2[1]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.scatter(df2['x'+str(i+1)+'1'],df2['x'+str(i+1)+'2'],s=200,c=df2['y'+str(i+1)],edgecolor='k')
    plt.grid(True)


# ## Random regression/classification problem generation using symbolic function

from Symbolic_regression_classification_generator import gen_regression_symbolic, gen_classification_symbolic


# ### Generate regression data with a symbolic expression of:
# $$\frac{x_1^2}{2}-3x_2+20.\text{sin}(x_3)$$

data8 = gen_regression_symbolic(m='((x1^2)/2-3*x2)+20*sin(x3)',n_samples=50,noise=0.01)
df8=pd.DataFrame(data8, columns=['x'+str(i) for i in range(1,4)]+['y'])

df8.head()

plt.figure(figsize=(18,5))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.scatter(df8[df8.columns[i-1]],df8['y'],s=200,c='orange',edgecolor='k')
    plt.grid(True)


# ### Generate regression data with a symbolic expression of:
# $$x_1^2*sin(x_1)$$

data8 = 0.1*gen_regression_symbolic(m='x1^2*sin(x1)',n_samples=200,noise=0.05)
df8=pd.DataFrame(data8, columns=['x'+str(i) for i in range(1,2)]+['y'])

plt.figure(figsize=(8,5))
plt.scatter(df8['x1'],df8['y'],s=100,c='orange',edgecolor='k')
plt.grid(True)


# ### Generate classification data with a symbolic expression of:
# $$\frac{x_1^2}{3}-\frac{x_2^2}{15}$$

data9 = gen_classification_symbolic(m='((x1^2)/3-(x2^2)/15)',n_samples=500,flip_y=0.01)
df9=pd.DataFrame(data9, columns=['x'+str(i) for i in range(1,3)]+['y'])

df9.head()

plt.figure(figsize=(8,5))
plt.scatter(df9['x1'],df9['x2'],c=df9['y'],s=100,edgecolors='k')
plt.xlabel('x1',fontsize=14)
plt.ylabel('x2',fontsize=14)
plt.grid(True)
plt.show()


# ### Generate classification data with a symbolic expression of:
# $$x_1-3.\text{sin}\frac{x_2}{2}$$

data9 = gen_classification_symbolic(m='x1-3*sin(x2/2)',n_samples=500,flip_y=0.01)
df9=pd.DataFrame(data9, columns=['x'+str(i) for i in range(1,3)]+['y'])

plt.figure(figsize=(8,5))
plt.scatter(df9['x1'],df9['x2'],c=df9['y'],s=100,edgecolors='k')
plt.xlabel('x1',fontsize=14)
plt.ylabel('x2',fontsize=14)
plt.grid(True)
plt.show()