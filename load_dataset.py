import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

class LoadDataset:
    def __init__(self, which, n_sample = 500, n_features = 20, generate = False):
        if which == 'bc':
            self.data = load_breast_cancer()
        elif which == 'hp':
            df = pd.read_csv("data/hepatitis.csv")
            #df = df.fillna(method='ffill')
            #df['Class'] = np.where(df['Class'] == 'Yes', 0, 1)
            feature_names = list(df.columns)
            feature_names = feature_names[:-1]
            target_names = np.array(['yes', 'no'])

            data = np.array(df.iloc[:, 0:19])
            target = np.array(df['Class'])

            self.data = Bunch(data=data, target=target,
                              target_names=target_names,
                              feature_names=feature_names)
        elif which == 'ildp':
            df = pd.read_csv("data/ildp.csv")
            df = df.fillna(method='ffill')
            df['class'] = np.where(df['class'] == 'Yes', 0, 1)
            feature_names = list(df.columns)
            feature_names = feature_names[:-1]
            target_names = np.array(['yes', 'no'])

            #data = preprocessing.scale(np.array(df.iloc[:, 0:9]))
            data = np.array(df.iloc[:, 0:9])
            target = np.array(df['class'])

            self.data = Bunch(data=data, target=target,
                  target_names=target_names,
                  feature_names=feature_names, dataframe = df.drop(columns=['class', 'id']))
        elif which == 'thyroid':
            df = pd.read_csv("data/thyroid-complete.csv")
            #df = pd.read_csv("data/thyroid/thyroid-train.csv")
            #df = pd.read_csv("data/thyroid/thyroid-test.csv")
            #df = df.fillna(method='ffill')
            #df['Class'] = np.where(df['Class'] == 'Yes', 0, 1)
            feature_names = list(df.columns)
            feature_names = feature_names[:-1]
            target_names = np.array(['normal', 'hyperthyroidism', 'hypothyroidism']) # 1, 2, 3 repectively

            data = np.array(df.iloc[:, 0:21])
            target = np.array(df['Class'])

            self.data = Bunch(data=data, target=target,
                              target_names=target_names,
                              feature_names=feature_names)

        elif which == 'synth':
            from sklearn.datasets import make_classification
            x, y = make_classification(n_samples=n_sample, n_features=n_features,
                                       n_informative=2, n_redundant=0, n_repeated=0,
                                       n_classes=2, n_clusters_per_class=2, weights=None,
                                       flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                                       scale=1.0, shuffle=True, random_state=0)


            df = pd.read_csv("data/ildp.csv")
            df = df.fillna(method='ffill')
            df['class'] = np.where(df['class'] == 'Yes', 0, 1)

            feature_names = ['x'+str(i) for i in range(1,n_features + 1)]
            target_names = np.array(['yes', 'no'])

            self.data = Bunch(data=x, target=y,
                  target_names=target_names,
                  feature_names=feature_names)

        elif which == 'synth1':
            X = np.zeros(shape=(500, 3), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3']
            y = []

            if generate:
                # Equation f1(x)
                x1 = np.random.uniform(0, 30, 500)
                x2 = np.random.uniform(0, 30, 500)
                x3 = np.random.uniform(0, 30, 500)

                #X = np.zeros(shape=(500, 3), dtype=np.float64)

                X[:, 0] = x1
                X[:, 1] = x2
                X[:, 2] = x3

                # feature_names = ['X1', 'X2', 'X3']
                #
                # y = []

                for i in range(0, 500):
                    if x1[i] <= 10:
                        if (x1[i] - (x2[i] * 4) + (x3[i] * 2) + 3) > 0:
                            y.append(1)
                        else:
                            y.append(0)
                    elif x1[i] > 10 and x1[i] <= 20:
                        if ((-2 * x1[i]) - (x2[i] * 3) + x2[i] - 2) > 0:
                            y.append(1)
                        else:
                            y.append(0)
                    else:
                        if ((x1[i] * 3) + x2[i] - (x3[i] * 2) + 2) > 0:
                            y.append(1)
                        else:
                            y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s1.npy")
                test = np.load("data/synthetic/X_test_s1.npy")
                labels_train = np.load("data/synthetic/y_train_s1.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s1.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test

            self.data = Bunch(data=X, target=y, feature_names=feature_names)

        elif which == 'synth2':
            # Equation f1(x)
            X = np.zeros(shape=(500, 10), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
            y = []

            if generate:
                x1 = np.random.uniform(0, 30, 500)
                x2 = np.random.uniform(0, 30, 500)
                x3 = np.random.uniform(0, 30, 500)

                X[:, 0] = x1
                X[:, 1] = x2
                X[:, 2] = x3
                X[:, 3] = np.random.uniform(0, 30, 500)
                X[:, 4] = np.random.uniform(0, 30, 500)
                X[:, 5] = np.random.uniform(0, 30, 500)
                X[:, 6] = np.random.uniform(0, 30, 500)
                X[:, 7] = np.random.uniform(0, 30, 500)
                X[:, 8] = np.random.uniform(0, 30, 500)
                X[:, 9] = np.random.uniform(0, 30, 500)


                for i in range(0, 500):
                    if x1[i] <= 10:
                        if (x1[i] - (x2[i] * 4) + (x3[i] * 2) + 3) > 0:
                            y.append(1)
                        else:
                            y.append(0)
                    elif x1[i] > 10 and x1[i] <= 20:
                        if ((-2 * x1[i]) - (x2[i] * 3) + x2[i] - 2) > 0:
                            y.append(1)
                        else:
                            y.append(0)
                    else:
                        if ((x1[i] * 3) + x2[i] - (x3[i] * 2) + 2) > 0:
                            y.append(1)
                        else:
                            y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s2.npy")
                test = np.load("data/synthetic/X_test_s2.npy")
                labels_train = np.load("data/synthetic/y_train_s2.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s2.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test

            self.data = Bunch(data=X, target=y, feature_names=feature_names)

        elif which == 'synth3':
            # Equation f2(x)
            X = np.zeros(shape=(500, 3), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3']

            y = []

            if generate:
                x1 = np.random.uniform(-100, 100, 500)
                x2 = np.random.uniform(-100, 100, 500)
                x3 = np.random.uniform(-100, 100, 500)

                X[:, 0] = x1
                X[:, 1] = x2
                X[:, 2] = x3


                for i in range(0, 500):
                    if ((x1[i] ** 3) - (2 * (x2[i] ** 2)) + (x3[i] * 3)) > 0:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s3.npy")
                test = np.load("data/synthetic/X_test_s3.npy")
                labels_train = np.load("data/synthetic/y_train_s3.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s3.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test

            self.data = Bunch(data=X, target=y, feature_names=feature_names)

        elif which == 'synth4':
            # Equation f3(x)
            X = np.zeros(shape=(500, 3), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3']

            y = []
            if generate:
                x1 = np.random.uniform(-10, 10, 500)
                x2 = np.random.uniform(-10, 10, 500)
                x3 = np.random.uniform(-10, 10, 500)

                X = np.zeros(shape=(500, 3), dtype=np.float64)

                X[:, 0] = x1
                X[:, 1] = x2
                X[:, 2] = x3

                feature_names = ['X1', 'X2', 'X3']

                y = []

                for i in range(0, 500):
                    if (x1[i] - (x2[i] * (math.sin(x2[i])) ** 2)) > 0:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s4.npy")
                test = np.load("data/synthetic/X_test_s4.npy")
                labels_train = np.load("data/synthetic/y_train_s4.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s4.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test

            self.data = Bunch(data=X, target=y, feature_names=feature_names)

        elif which == 'synth5':
            # Equation f4(x)
            X = np.zeros(shape=(500, 9), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
            y = []

            if generate:
                x1 = np.random.randint(1, 5, 500)
                x2 = np.random.randint(0, 2, 500)
                x3 = np.random.randint(0, 2, 500)
                x4 = np.random.randint(0, 2, 500)
                x5 = np.random.randint(0, 2, 500)
                x6 = np.random.randint(0, 2, 500)
                x7 = np.random.randint(0, 2, 500)
                x8 = np.random.randint(0, 2, 500)
                x9 = np.random.randint(0, 2, 500)

                X[:, 0] = x1
                X[:, 1] = x2
                X[:, 2] = x3
                X[:, 3] = x4
                X[:, 4] = x5
                X[:, 5] = x6
                X[:, 6] = x7
                X[:, 7] = x8
                X[:, 8] = x9

                for i in range(0, 500):
                    condition = ((x1[i] == 1 and x2[i] and x3[i]) or
                                 (x1[i] == 2 and x4[i] and x5[i]) or
                                 (x1[i] == 3 and x6[i] and x7[i]) or
                                 (x1[i] == 4 and x8[i] and x9[i]))
                    if condition:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s5.npy")
                test = np.load("data/synthetic/X_test_s5.npy")
                labels_train = np.load("data/synthetic/y_train_s5.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s5.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test
            self.data = Bunch(data=X, target=y, feature_names=feature_names)

        elif which == 'synth6':
            # Equation f4(x)
            X = np.zeros(shape=(500, 20), dtype=np.float64)
            feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
                             'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15',
                             'X16', 'X17', 'X18', 'X19', 'X20']

            y = []

            if generate:
                x1 = np.random.randint(1, 5, 500)
                X[:, 0] = x1

                for ind in range(1,20):
                    X[:, ind] = np.random.randint(0, 2, 500)

                x2 = X[:, 1]
                x3 = X[:, 2]
                x4 = X[:, 3]
                x5 = X[:, 4]
                x6 = X[:, 5]
                x7 = X[:, 6]
                x8 = X[:, 7]
                x9 = X[:, 8]

                for i in range(0, 500):
                    condition = ((x1[i] == 1 and x2[i] and x3[i]) or
                                 (x1[i] == 2 and x4[i] and x5[i]) or
                                 (x1[i] == 3 and x6[i] and x7[i]) or
                                 (x1[i] == 4 and x8[i] and x9[i]))
                    if condition:
                        y.append(1)
                    else:
                        y.append(0)
            else:
                train = np.load("data/synthetic/X_train_s6.npy")
                test = np.load("data/synthetic/X_test_s6.npy")
                labels_train = np.load("data/synthetic/y_train_s6.npy").tolist()
                labels_test = np.load("data/synthetic/y_test_s6.npy").tolist()
                X = np.row_stack([train, test])

                y = labels_train + labels_test

            self.data = Bunch(data=X, target=y, feature_names=feature_names)