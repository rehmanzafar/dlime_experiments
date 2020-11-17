import pandas as pd
import numpy as np

df = pd.read_csv("results/csv/rf_lime_jdist_hp.csv")
feature_names = list(df.columns)
data = np.array(df.iloc[:, 0:9])

m = data.mean()

print(m*100)