from sklearn.model_selection import train_test_split

from load_dataset import LoadDataset
import numpy as np

test = LoadDataset(which='synth3', generate= True)
X = test.data.data
feature_names = test.data.feature_names
#target_names = test.data.target_names
target_names = np.array(['Yes', 'No'])
train, test, labels_train, labels_test = train_test_split(test.data.data, test.data.target, train_size=0.80)

np.save("data/synthetic/X_train_s3_exp.npy", train)
np.save("data/synthetic/X_test_s3_exp.npy", test)
np.save("data/synthetic/y_train_s3_exp.npy", labels_train)
np.save("data/synthetic/y_test_s3_exp.npy", labels_test)