import string
import matplotlib.pyplot as plt
import numpy as np
from utils import InvalidExplanationMode


def id_generator(size=15, random_state=None):
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


class DomainMapper(object):
    def __init__(self):
        pass

    def map_exp_ids(self, exp, **kwargs):
        return exp


class Explanation(object):

    def __init__(self,
                 domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.local_exp = {}
        self.intercept = {}
        self.score = {}
        self.local_pred = {}
        self.scaled_data = None
        self.easy_model_coef = {}
        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1
        else:
            raise InvalidExplanationMode('Invalid explanation mode "{}". '
                            'Should be either "classification" '
                            'or "regression".'.format(mode))

    def available_labels(self):
        try:
            assert self.mode == "classification"
        except AssertionError:
            raise NotImplementedError('Not supported for regression explanations.')
        else:
            ans = self.top_labels if self.top_labels else self.local_exp.keys()
            return list(ans)

    def as_list_one(self, label=1, **kwargs):
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_list_zero(self, label=0, **kwargs):
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_map(self):
        return self.local_exp

    def as_pyplot_figure(self, label=0, type='h', **kwargs):
        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        if type == 'h':
            plt.barh(pos, vals, align='center', color=colors)
            plt.yticks(pos, names)
        else:
            plt.bar(pos, vals, align='center', color=colors)
            plt.xticks(pos, names, rotation=90)

        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        return fig, names

    def as_pyplot_to_figure(self, type='h', name = None, label='0', lp=None, **kwargs):
        if label == '0':
            exp = self.as_list_zero(label=0, **kwargs)
        else:
            exp = self.as_list_one(label=1, **kwargs)
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        #pos = np.arange(len(exp)) + .2
        pos = np.arange(len(exp)) / 3

        fig = plt.figure(figsize=(4, 2))
        if type == 'h':
            plt.barh(pos, vals, align='center', color=colors, height=0.2)
            plt.yticks(pos, names)
        else:
            plt.bar(pos, vals, align='center', color=colors)
            plt.xticks(pos, names, rotation=90)

        if self.mode == "classification":
            title = 'Local explanation for class: %s' % self.class_names[int(label)]
        else:
            title = 'Local explanation'
        plt.title(title)
        #plt.suptitle('Sup title', y=1.05, fontsize=18)
        #plt.savefig(str(name) + ".png")
        filename= 'results/' + str(name)+".pdf"
        plt.savefig(filename, bbox_inches='tight')
        return fig, names