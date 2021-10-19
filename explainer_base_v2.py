import numpy as np
import shap as shap
from treeinterpreter import treeinterpreter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, lars_path, LinearRegression
from sklearn.utils import check_random_state

from plots.summary import summary_plot


class LimeBase(object):
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights, num_features, n_method)
        elif method == 'boruta':
            rf = RandomForestRegressor(n_jobs=-1)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
            feat_selector.fit(data, labels)
            feat_selector.ranking_
            return np.where(feat_selector.support_)[0]

    def explain_instance_with_data(self,
        neighborhood_data,
        subspace_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection = 'auto',
        model_regressor = None,
        regressor = 'tree',
        fidelity = True,
        bmodel = None):



        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        elif model_regressor is 'tree':
            model_regressor = RandomForestRegressor(n_estimators= 10, random_state=0) #estimators = 10, max_depth=10, max_features=10
        else:
            model_regressor = LinearRegression()
        easy_model = model_regressor
        easy_model.fit(neighborhood_data, labels_column, sample_weight=weights)

        #explainer = shap.TreeExplainer(easy_model)
        #shap_values = explainer.shap_values(neighborhood_data[0, used_features].reshape(1, -1))

        #shap.summary_plot(shap_values, neighborhood_data[0, used_features].reshape(1, -1))

        prediction_score = easy_model.score(
            neighborhood_data,
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])

        if regressor is 'tree':
            #prediction ≈ bias + feature_contributions
            local_pred, bias, contrib = treeinterpreter.predict(easy_model, neighborhood_data[0].reshape(1, -1))
            # summary_plot(contrib,
            #              neighborhood_data[0].reshape(1, -1)
            #              #, used_features
            #              #,plot_type='bar'
            #              )
            return (bias,
                    sorted(zip(used_features, contrib[0]),
                           key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred,contrib[0], easy_model, used_features,
                weights)
        else:
            return (easy_model.intercept_,
                        sorted(zip(used_features, easy_model.coef_),
                               key=lambda x: np.abs(x[1]), reverse=True),
                        prediction_score, local_pred, easy_model.coef_, easy_model, used_features,
                weights)