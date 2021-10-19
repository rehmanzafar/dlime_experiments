import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from explainer_tabular import LimeTabularExplainer
from load_dataset import LoadDataset

test = LoadDataset(which='synth2')
X = test.data.data
feature_names = test.data.feature_names
#target_names = test.data.target_names
target_names = np.array(['Yes', 'No'])
#train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)
# np.save("data/synthetic/X_train_s5.npy", train)
# np.save("data/synthetic/X_test_s5.npy", test)
# np.save("data/synthetic/y_train_s5.npy", labels_train)
# np.save("data/synthetic/y_test_s5.npy", labels_test)

# fix it
# train = np.load("data/X_train.npy")
# test = np.load("data/X_test.npy")
# labels_train = np.load("data/y_train.npy")
# labels_test = np.load("data/y_test.npy")

train = np.load("data/synthetic/X_train_s2.npy")
test = np.load("data/synthetic/X_test_s2.npy")
labels_train = np.load("data/synthetic/y_train_s2.npy")
labels_test = np.load("data/synthetic/y_test_s2.npy")

#rf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=5, max_features=5)
#rf.fit(train, labels_train)
#mean_accuracy = rf.score(test, labels_test)

lgr = LogisticRegression(max_iter=1000, random_state=0).fit(train, labels_train)
mean_accuracy = lgr.score(test, labels_test)
e_true = lgr.coef_

explainer = LimeTabularExplainer(train,
                                 mode="classification",
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 feature_selection="none",
                                 discretize_continuous=True,
                                 verbose=False)

# clustering = AgglomerativeClustering().fit(X)
# names = list(feature_names)+["membership"]
# clustered_data = np.column_stack([X, clustering.labels_])

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

list_avg_cos_sim_dlime = []
list_avg_cos_sim_lime = []
lst_fid_dlime = []
for x in range(0, test.shape[0]):
    use_case_one_features = []
    use_case_two_features = []
    use_case_three_features = []
    use_case_four_features = []
    dlime_list_coef = []
    #dlime_list_coef_1 = []
    lime_list_coef = []
    #lime_list_coef_1 = []

    dlime_fid_list =[]

    subset = train[indices[x], :] # for NN

    # p_label = clabel[indices[x]] # for HC
    # N = clustered_data[clustered_data[:, 9] == p_label]
    # subset = np.delete(N, 9, axis=1)
    for i in range(0, 10):
        exp_dlime, fid_dlime = explainer.explain_instance_hclust(test[x],
                                             lgr.predict_proba,
                                             num_features=test.shape[1],
                                             model_regressor=LinearRegression(),
                                             clustered_data = subset,
                                             regressor = 'linear',
                                             explainer='dlime',
                                             labels=(0, 1),
                                             # labels=(0,1,2),
                                             # labels=(0,1,2,3,4,4,5,6,7,8,9),
                                             blmodel = lgr,
                                             fidelity = True
                                             )

        #fig_dlime, r_features = exp_dlime.as_pyplot_to_figure(type='h', name = i+.2, label='0')
        #fig_dlime.show()
        #use_case_two_features.append(r_features)
        dlime_list_coef.append(exp_dlime.easy_model_coef[0])
        dlime_fid_list.append(fid_dlime)
        #dlime_list_coef_0.append(exp_dlime.easy_model_coef[0])
        #dlime_list_coef_1.append(exp_dlime.easy_model_coef[1])


        # exp_lime = explainer.explain_instance_hclust(test[x],
        #                                      lgr.predict_proba,
        #                                      num_features=test.shape[1],
        #                                      model_regressor= LinearRegression(),
        #                                      regressor = 'linear',
        #                                      explainer = 'lime', labels=(0,1))

        #fig_lime, r_features = exp_lime.as_pyplot_to_figure(type='h', name = i+.3, label='0')
        #fig_lime.show()
        #use_case_three_features.append(r_features)

        #lime_list_coef.append(exp_lime.easy_model_coef[0])

        #lime_list_coef_0.append(exp_lime.easy_model_coef[0])
        #lime_list_coef_0.append(exp_lime.easy_model_coef[1])


    ################################################
    # sim = jaccard_distance(use_case_two_features)
    # #np.savetxt("results/rf_dlime_jdist_bc.csv", sim, delimiter=",")
    # print(np.asarray(sim).mean())
    #
    # plt.matshow(sim);
    # plt.colorbar()
    # #plt.savefig("results/sim_use_case_2.pdf", bbox_inches='tight')
    # plt.show()

    ################################################
    # sim = jaccard_distance(use_case_three_features)
    # #np.savetxt("results/rf_lime_jdist_bc.csv", sim, delimiter=",")
    # print(np.asarray(sim).mean())
    #
    # plt.matshow(sim);
    # plt.colorbar()
    # #plt.savefig("results/sim_use_case_3.pdf", bbox_inches='tight')
    # plt.show()

    #e_pred_0 = exp_dlime.easy_model_coef[0]
    #e_pred_1 = exp_dlime.easy_model_coef[1]

    #ext_0 = np.expand_dims(e_pred_0, axis=0)
    #ext_1 = np.expand_dims(e_pred_1, axis=0)


    #cos_similarity_0 = cosine_similarity(e_true, ext_0)
    #cos_similarity_1 = cosine_similarity(e_true, ext_1)

    lst_fid_dlime.append(sum(dlime_fid_list) / len(dlime_fid_list))

    cos_similarity_dlime = abs(cosine_similarity(e_true, dlime_list_coef))
    #cos_similarity_lime = abs(cosine_similarity(e_true, lime_list_coef))

    avg_cos_similarity_dlime = np.mean(cos_similarity_dlime)
    #avg_cos_similarity_lime = np.mean(cos_similarity_lime)

    list_avg_cos_sim_dlime.append(avg_cos_similarity_dlime)
    #list_avg_cos_sim_lime.append(avg_cos_similarity_lime)

    print(f"Actuale Label = {labels_test[x]}")
    print(f"Cosine Similarity DLIME-NN after 10 iterations = {avg_cos_similarity_dlime}")
    print(f"Fidelity DLIME-NN after 10 iterations = {sum(dlime_fid_list)/len(dlime_fid_list)}")
    #print(f"Cosine Similarity LIME after 10 iterations = {avg_cos_similarity_lime}")
o_avg_cos_similarity_dlime = np.mean(np.array(list_avg_cos_sim_dlime))
#o_avg_cos_similarity_lime = np.mean(np.array(list_avg_cos_sim_lime))
print(f"Cosine Similarity DLIME-NN Overall = {o_avg_cos_similarity_dlime}")
avg_dlime_fid = sum(lst_fid_dlime)/len(lst_fid_dlime)
print(f"Fidelity DLIME-NN Overall = {avg_dlime_fid}")

#print(f"Cosine Similarity LIME Overall = {o_avg_cos_similarity_lime}")
print("Execution done")