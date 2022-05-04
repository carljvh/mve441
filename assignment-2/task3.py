import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feat = pd.read_csv('assignment-2/data/data.csv', index_col=0)
feat = VarianceThreshold(threshold=2.2343).fit_transform(feat)
feat = StandardScaler().fit_transform(feat)

n_PC = 801

pca = PCA(n_components=n_PC)
principalComponents = pca.fit_transform(feat)
principalDf = pd.DataFrame(data = principalComponents)

principalDf.plot.scatter(x=0, y=1)
plt.savefig('assignment-2/IMGs/task3/prin_comp_01.png')
plt.clf()

principalDf.plot.scatter(x=1, y=2)
plt.savefig('assignment-2/IMGs/task3/prin_comp_12.png')
plt.clf()

principalDf.plot.scatter(x=2, y=3)
plt.savefig('assignment-2/IMGs/task3/prin_comp_23.png')
plt.clf()

principalDf.plot.scatter(x=3, y=4)
plt.savefig('assignment-2/IMGs/task3/prin_comp_34.png')
plt.clf()

principalDf.plot.scatter(x=4, y=5)
plt.savefig('assignment-2/IMGs/task3/prin_comp_45.png')
plt.clf()

principalDf.plot.scatter(x=5, y=6)
plt.savefig('assignment-2/IMGs/task3/prin_comp_56.png')
plt.clf()

principalDf.plot.scatter(x=6, y=7)
plt.savefig('assignment-2/IMGs/task3/prin_comp_67.png')
plt.clf()

principalDf.plot.scatter(x=7, y=8)
plt.savefig('assignment-2/IMGs/task3/prin_comp_78.png')
plt.clf()

princ_var = principalDf.var().tolist()
plt.plot(np.arange(n_PC), princ_var, 'o')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.savefig('assignment-2/IMGs/task3/scree.png')
plt.clf()


for idx, v in enumerate(princ_var):
    if v < 1:
        print('Index of PC with variance less then 1: %i' % (idx))
        principalDf.iloc[:,0:idx].to_csv('assignment-2/data/pc_521.csv')
        break
