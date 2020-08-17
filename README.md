# DLIME: A Deterministic Local Interpretable Model-Agnostic Explanations Approach for Computer Aided Diagnostic Systems
## Experiments

### Setup Environment
The following python environment and packages are used to conduct the experiments:

*  python==3.6
*  Boruta==0.1.5
*  numpy==1.16.1
*  pandas==0.24.2
*  scikit-learn==0.20.2
*  scipy==1.2.1

These packages can be installed by executing the following command: ``pip3.6 install -r requirements.txt``

### Dataset
To conduct the experiments we have used the following three healthcare datasets from UCI repository:

*  [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
*  [Hepatitis Dataset](https://archive.ics.uci.edu/ml/datasets/hepatitis)
*  [Indian Liver Patient Dataset](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))

Breast cancer dataset comes along with scikit-learn package, therefore, there is no need to download this dataset. The rest of the datasets are already download and available inside the "data" folder in csv format.

### Algorithms
The following classifiers and algorithms are used in this study:

*  Random Forest
*  Neural Networks
*  Linear Regression
*  K-Nearest Neighbours
*  Agglomerative Hierarchical Clustering

In the experiments, 80% data is used for training and the remaining 20% data is used for testing. Further, the Random Forest, Neural Networks and KNN classifiers are trained with the following parameters:

```
RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5, max_features=5)

MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

NearestNeighbors(n_neighbors=1, algorithm='ball_tree')

AgglomerativeClustering(n_clusters=2, affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward', memory=None)
```

### Execute Code
Run the following files to reproduce the results. The results of LIME are not deterministic and it may produce different results.
 
##### Experiments on Breast Cancer Dataset:

*  python3.6 experiments_bc_nn.py
*  python3.6 experiments_bc_rf.py


##### Experiments on Indian Liver Patient Dataset:

*  python3.6 experiments_ildp_nn.py
*  python3.6 experiments_ildp_rf.py


##### Experiments on Hepatitis Dataset:

*  python3.6 experiments_hp_nn.py
*  python3.6 experiments_hp_rf.py


### Results
The results will be saved inside "results" directory in pdf and csv format.

### Citation
Please consider citing our work if you use this code for your research.
```
@InProceedings{zafar2019dlime,
  author    = {Muhammad Rehman Zafar and Naimul Mefraz Khan},
  title     = {DLIME: A Deterministic Local Interpretable Model-Agnostic Explanations Approach for Computer-Aided Diagnosis Systems},
  booktitle = {In proceeding of ACM SIGKDD Workshop on Explainable AI/ML (XAI) for Accountability, Fairness, and Transparency},
  year      = {2019},
  publisher = {ACM},
  address   = {Anchorage, Alaska}
}
```
