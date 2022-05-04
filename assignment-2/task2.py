import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

feat = pd.read_csv('assignment-2/data/data.csv', index_col=0)

sel = VarianceThreshold(threshold=2.2343)
feat_red = sel.fit_transform(feat)

feat = pd.DataFrame(feat_red)

feat.mean(axis=0).plot(kind='bar')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlabel('features')
plt.ylabel('mean')
plt.savefig('assignment-2/IMGs/task2/mean_red.png')
plt.clf()

feat.std(axis=0).plot(kind='bar')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlabel('features')
plt.ylabel('standard deviation')
plt.savefig('assignment-2/IMGs/task2/std_red.png')
plt.clf()

# doesn't work, covariance equal to 1? cant invert covariance matrix?
feat.plot.kde(legend=False)
plt.savefig('assignment-2/IMGs/task2/kde_red.png')
plt.clf()

