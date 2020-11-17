import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from load_dataset import LoadDataset

test = LoadDataset(which='synth3')

X = test.data.data
y = test.data.target
feature_names = test.data.feature_names

df = pd.DataFrame(data=X, columns=feature_names)
df['Class'] = y

sn.heatmap(df.corr(), annot=True)
plt.show()


# plt.style.use('ggplot')
#
# plt.imshow(df.corr(), cmap=plt.cm.Reds, interpolation='nearest')
# plt.colorbar()
# tick_marks = [i for i in range(len(df.columns))]
# plt.xticks(tick_marks, df.columns, rotation='vertical')
# plt.yticks(tick_marks, df.columns)
# plt.show()