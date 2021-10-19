import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset

test = LoadDataset(which='mc2')
X = test.data.data
XX = test.data.data
feature_names = test.data.feature_names
target_names = test.data.target_names
#target_names = np.array(['Yes', 'No'])

#train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)
# np.save("data/synthetic/X_train_s5.npy", train)
# np.save("data/synthetic/X_test_s5.npy", test)
# np.save("data/synthetic/y_train_s5.npy", labels_train)
# np.save("data/synthetic/y_test_s5.npy", labels_test)

# fix it
train = np.load("data/mc/X_train_mc_cardio.npy")
test = np.load("data/mc/X_test_mc_cardio.npy")
labels_train = np.load("data/mc/y_train_mc_cardio.npy")
labels_test = np.load("data/mc/y_test_mc_cardio.npy")

# train = np.load("data/synthetic/X_train_s6.npy")
# test = np.load("data/synthetic/X_test_s6.npy")
# labels_train = np.load("data/synthetic/y_train_s6.npy")
# labels_test = np.load("data/synthetic/y_test_s6.npy")

#rf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5, max_features=5)
#rf.fit(train, labels_train)
#mean_accuracy = rf.score(test, labels_test)

lgr = LogisticRegression(random_state=0).fit(train, labels_train)
mean_accuracy = lgr.score(test, labels_test)
e_true = lgr.coef_

explainer = LimeTabularExplainer(train,
                                 mode="classification",
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 feature_selection="none",
                                 discretize_continuous=True,
                                 verbose=False)

clustering = AgglomerativeClustering().fit(X)
names = list(feature_names)+["membership"]
m_index = len(names)-1
clustered_data = np.column_stack([X, clustering.labels_])

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train)
distances, indices = nbrs.kneighbors(test)
clabel = clustering.labels_

nn_nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(train)
nn_distances, nn_indices = nn_nbrs.kneighbors(test)

kmeans = KMeans(n_clusters=2, random_state=0).fit(XX)
#names = list(feature_names)+["membership"]
#m_index = len(names) - 1
km_clustered_data = np.column_stack([XX, kmeans.labels_])

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))
def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1-jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim

list_avg_cos_sim_dlime = []
list_avg_cos_sim_dlime_tree = []
list_avg_cos_sim_dlime_nn = []
list_avg_cos_sim_dlime_km = []
list_avg_cos_sim_lime = []
for x in range(0, test.shape[0]):
    dlime_list_coef = []
    dlime_tree_list_coef = []
    dlime_nn_list_coef = []
    dlime_km_list_coef = []
    lime_list_coef = []

    subset_nn = train[nn_indices[x], :] # for NN

    p_label = clabel[indices[x]]
    N = clustered_data[clustered_data[:, m_index] == p_label]
    subset = np.delete(N, m_index, axis=1)

    p_label_km = kmeans.predict([test[x]])
    NN = clustered_data[clustered_data[:, m_index] == p_label_km[0]]
    subset_km = np.delete(NN, m_index, axis=1)

    for i in range(0, 10):

        ## DLIME - Linear
        exp_dlime = explainer.explain_instance_hclust(test[x],
                                             lgr.predict_proba,
                                             num_features=test.shape[1],
                                             model_regressor=LinearRegression(),
                                             clustered_data = subset,
                                             regressor = 'linear',
                                             explainer='dlime',
                                             labels=(0,1))

        dlime_list_coef.append(exp_dlime.easy_model_coef[0])

        ## DLIME-Tree
        exp_dlime_tree = explainer.explain_instance_hclust(test[x],
                                                                 lgr.predict_proba,
                                                                 num_features=test.shape[1],
                                                                 model_regressor="tree",
                                                                 clustered_data=subset,
                                                                 regressor='tree',
                                                                 explainer='dlime', labels=(0, 1))
        dlime_tree_list_coef.append(exp_dlime_tree.easy_model_coef[0])

        ## DLIME - NN
        exp_dlime_nn = explainer.explain_instance_hclust(test[x],
                                                      lgr.predict_proba,
                                                      num_features=test.shape[1],
                                                      model_regressor=LinearRegression(),
                                                      clustered_data=subset_nn,
                                                      regressor='linear',
                                                      explainer='dlime', labels=(0, 1))
        dlime_nn_list_coef.append(exp_dlime_nn.easy_model_coef[0])

        ## DLIME - KM
        exp_dlime_km = explainer.explain_instance_hclust(test[x],
                                                                 lgr.predict_proba,
                                                                 num_features=test.shape[1],
                                                                 model_regressor=LinearRegression(),
                                                                 clustered_data=subset_km,
                                                                 regressor='linear',
                                                                 explainer='dlime', labels=(0, 1))
        dlime_km_list_coef.append(exp_dlime_km.easy_model_coef[0])

        ## LIME
        exp_lime = explainer.explain_instance_hclust(test[x],
                                             lgr.predict_proba,
                                             num_features=test.shape[1],
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear',
                                             explainer = 'lime', labels=(0,1))


        lime_list_coef.append(exp_lime.easy_model_coef[0])

    cos_similarity_dlime = abs(cosine_similarity(e_true, dlime_list_coef))
    cos_similarity_dlime_tree = abs(cosine_similarity(e_true, dlime_tree_list_coef))
    cos_similarity_dlime_nn = abs(cosine_similarity(e_true, dlime_nn_list_coef))
    cos_similarity_dlime_km = abs(cosine_similarity(e_true, dlime_km_list_coef))
    cos_similarity_lime = abs(cosine_similarity(e_true, lime_list_coef))

    avg_cos_similarity_dlime = np.mean(cos_similarity_dlime)
    avg_cos_similarity_dlime_tree = np.mean(cos_similarity_dlime_tree)
    avg_cos_similarity_dlime_nn = np.mean(cos_similarity_dlime_nn)
    avg_cos_similarity_dlime_km = np.mean(cos_similarity_dlime_km)
    avg_cos_similarity_lime = np.mean(cos_similarity_lime)

    list_avg_cos_sim_dlime.append(avg_cos_similarity_dlime)
    list_avg_cos_sim_dlime_tree.append(avg_cos_similarity_dlime_tree)
    list_avg_cos_sim_dlime_nn.append(avg_cos_similarity_dlime_nn)
    list_avg_cos_sim_dlime_km.append(avg_cos_similarity_dlime_km)
    list_avg_cos_sim_lime.append(avg_cos_similarity_lime)

    #print(f"Actual Label = {labels_test[x]}")
    print(f"Cosine Similarity DLIME after 10 iterations = {avg_cos_similarity_dlime}")
    print(f"Cosine Similarity DLIME-Tree after 10 iterations = {avg_cos_similarity_dlime_tree}")
    print(f"Cosine Similarity DLIME-NN after 10 iterations = {avg_cos_similarity_dlime_nn}")
    print(f"Cosine Similarity DLIME-KM after 10 iterations = {avg_cos_similarity_dlime_km}")
    print(f"Cosine Similarity LIME after 10 iterations = {avg_cos_similarity_lime}")
o_avg_cos_similarity_dlime = np.mean(np.array(list_avg_cos_sim_dlime))
o_avg_cos_similarity_dlime_tree = np.mean(np.array(list_avg_cos_sim_dlime_tree))
o_avg_cos_similarity_dlime_nn = np.mean(np.array(list_avg_cos_sim_dlime_nn))
o_avg_cos_similarity_dlime_km = np.mean(np.array(list_avg_cos_sim_dlime_km))
o_avg_cos_similarity_lime = np.mean(np.array(list_avg_cos_sim_lime))
print(f"Cosine Similarity DLIME Overall = {o_avg_cos_similarity_dlime}")
print(f"Cosine Similarity DLIME-Tree Overall = {o_avg_cos_similarity_dlime_tree}")
print(f"Cosine Similarity DLIME-NN Overall = {o_avg_cos_similarity_dlime_nn}")
print(f"Cosine Similarity DLIME-KM Overall = {o_avg_cos_similarity_dlime_km}")
print(f"Cosine Similarity LIME Overall = {o_avg_cos_similarity_lime}")

from scipy import stats
# t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime, list_avg_cos_sim_lime)
# print("DLIME-Linear VS LIME t2:\t", t2, "p2:\t", p2)
#
# t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime, list_avg_cos_sim_dlime_tree)
# print("DLIME-Linear VS DLIME-TREE t2:\t", t2, "p2:\t", p2)
#
# t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime, list_avg_cos_sim_dlime_nn)
# print("DLIME-Linear VS DLIME-NN t2:\t", t2, "p2:\t", p2)
#
# t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime, list_avg_cos_sim_dlime_km)
# print("DLIME-Linear VS DLIME-KM t2:\t", t2, "p2:\t", p2)

# from scipy import stats
t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime_tree, list_avg_cos_sim_lime)
print("DLIME-Tree VS LIME t2:\t", t2, "p2:\t", p2)

t2, p2 = stats.ttest_ind(list_avg_cos_sim_dlime_km, list_avg_cos_sim_lime)
print("LIME VS DLIME-KM t2:\t", t2, "p2:\t", p2)

t2, p2 = stats.ttest_ind(list_avg_cos_sim_lime, list_avg_cos_sim_dlime_nn)
print("LIME VS DLIME-NN t2:\t", t2, "p2:\t", p2)

t2, p2 = stats.ttest_ind(list_avg_cos_sim_lime, list_avg_cos_sim_dlime)
print("LIME VS DLIME t2:\t", t2, "p2:\t", p2)

print("Execution done")
