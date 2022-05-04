import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feat = pd.read_csv('assignment-2/data/data.csv', index_col=0)
lab = pd.read_csv('assignment-2/data/labels.csv', index_col=0)

n_null_feat = feat.isnull().sum().sum()
n_null_lab = lab.isnull().sum()

# Missing values are coded as NaN
print('Number of NaN:\tfeatures: %i\tlabels: %i' % (n_null_feat, n_null_lab) )

feat.mean(axis=0).plot(kind='bar')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlabel('features')
plt.ylabel('mean')
plt.savefig('assignment-2/IMGs/task1/mean.png')
plt.clf()

# Some constant features can be observed (std = 0)
feat.std(axis=0).plot(kind='bar')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlabel('features')
plt.ylabel('standard deviation')
plt.savefig('assignment-2/IMGs/task1/std.png')
plt.clf()

# singular matrix: square but not invertible <=> determinant = 0
#feat.plot.kde()
#plt.show()
# plt.savefig('kde.png')
# plt.clf()