# Deterministic Local Interpretable Model-Agnostic Explanations for Stable Explainability
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

### Datasets
To conduct the experiments we have used the following three healthcare datasets from UCI repository:

*  [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
*  [Hepatitis](https://archive.ics.uci.edu/ml/datasets/hepatitis)
*  [Indian Liver Patient](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))
*  [Cardiotocography](https://archive.ics.uci.edu/ml/datasets/Cardiotocography)
*  [Thyroid](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)
*  [Handwritten Digits](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)

Breast cancer dataset comes along with scikit-learn package, therefore, there is no need to download this dataset. The rest of the datasets are already downloaded and available in "data" folder.

### Algorithms
The following classifiers and algorithms are used in this study:

*  Random Forest
*  Neural Networks
*  Linear Regression
*  Logistic Regression
*  K-Nearest Neighbours
*  K-Means Clustering
*  Agglomerative Hierarchical Clustering

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

##### For the quality of the explanations:

*  python3.6 experiments_bc_lgr_fidelity_v2p0-mc-v2.py
*  python3.6 evaluate_quality_v0.py


### Results
The results will be saved inside "results" directory in pdf and csv format. The quality of the explanation is shown in the image below:
![Quality of Explanations](https://github.com/rehmanzafar/dlime_experiments/blob/ongoing-experiment/results/quality_of_explanations.PNG)

### Citation
Please consider citing our work if you use this code for your research.
#### Initial Results
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
#### Extended Version
```
@article{zafar2021deterministic,
  title={Deterministic Local Interpretable Model-Agnostic Explanations for Stable Explainability},
  author={Zafar, Muhammad Rehman and Khan, Naimul},
  journal={Machine Learning and Knowledge Extraction},
  volume={3},
  number={3},
  pages={525--541},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
