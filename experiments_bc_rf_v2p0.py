import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset

test = LoadDataset(which='bc')
X = test.data.data
feature_names = test.data.feature_names
target_names = test.data.target_names

# train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)
# np.save("X_train.npy", train)
# np.save("X_test.npy", test)
# np.save("y_train.npy", labels_train)
# np.save("y_test.npy", labels_test)

train = np.load("data/X_train.npy")
test = np.load("data/X_test.npy")
labels_train = np.load("data/y_train.npy")
labels_test = np.load("data/y_test.npy")

rf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5, max_features=5)
rf.fit(train, labels_train)
mean_accuracy = rf.score(test, labels_test)

explainer = LimeTabularExplainer(train,
                                 mode="classification",
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 discretize_continuous=True,
                                 verbose=False)

#clustering = AgglomerativeClustering().fit(X)
#names = list(feature_names)+["membership"]
#clustered_data = np.column_stack([X, clustering.labels_])

nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(train)
distances, indices = nbrs.kneighbors(test)
#clabel = clustering.labels_

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


for x in range(0, test.shape[0]):
    use_case_one_features = []
    use_case_two_features = []
    use_case_three_features = []
    use_case_four_features = []
    subset = train[indices[x], :]
    for i in range(0, 10):
        #p_label = clabel[indices[x]]
        #subset = train[indices[x], :]
        #subset = np.delete(N, 30, axis=1)

        exp_dlime = explainer.explain_instance_hclust(test[x],
                                             rf.predict_proba,
                                             num_features=10,
                                             model_regressor=LinearRegression(),
                                             clustered_data = subset,
                                             regressor = 'linear', explainer='dlime', labels=(0,1))

        fig_dlime, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = i+.2, label='0')
        #fig_dlime.show()
        use_case_two_features.append(r_features)


        exp_lime = explainer.explain_instance_hclust(test[x],
                                             rf.predict_proba,
                                             num_features=10,
                                             model_regressor= LinearRegression(),
                                             regressor = 'linear', explainer = 'lime', labels=(0,1))

        fig_lime, r_features = exp_lime.as_pyplot_to_figure(type='h', name = i+.3, label='0')
        #fig_lime.show()
        use_case_three_features.append(r_features)


    ################################################
    sim = jaccard_distance(use_case_two_features)
    #np.savetxt("results/rf_dlime_jdist_bc.csv", sim, delimiter=",")
    print(np.asarray(sim).mean())

    plt.matshow(sim);
    plt.colorbar()
    #plt.savefig("results/sim_use_case_2.pdf", bbox_inches='tight')
    plt.show()

    ################################################
    sim = jaccard_distance(use_case_three_features)
    #np.savetxt("results/rf_lime_jdist_bc.csv", sim, delimiter=",")
    print(np.asarray(sim).mean())

    plt.matshow(sim);
    plt.colorbar()
    #plt.savefig("results/sim_use_case_3.pdf", bbox_inches='tight')
    plt.show()
