import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

n_fig = 4
E = 1
N = [2, 3, 4, 5, 6, 7, 8]
colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

cwd = os.getcwd()

features = pd.read_csv(cwd + '/assignment-2/data/data.csv', index_col=0).to_numpy()
features_variancefiltered = VarianceThreshold(threshold=2.2343).fit_transform(features)
features_variancefiltered_normalized = StandardScaler().fit_transform(features_variancefiltered)
pca = PCA(n_components=10)
principal_components = pca.fit_transform(features_variancefiltered_normalized)
X = pd.DataFrame(data = principal_components)
X.to_csv(cwd + '/assignment-2/data/pc_task6.csv')

features = pd.read_csv(cwd + '/assignment-2/data/pc_task6.csv', index_col=0).to_numpy()
labels = pd.read_csv(cwd + '/assignment-2/data/labels.csv').drop(columns='Unnamed: 0')

n_points = 50
eps = np.arange(1, n_points+1, 1)
min_samples = np.arange(1, n_points+1, 1)

sh_score = np.zeros((2, n_points, n_points))
ch_score = np.zeros((2, n_points, n_points))
db_score = np.ones((2, n_points, n_points))

for i, e in enumerate(eps):
    for j, ms in enumerate(min_samples):
        dbscan = DBSCAN(eps = e, min_samples = ms).fit(features)
        idx = np.where(dbscan.labels_ == -1)
        lab = np.delete(dbscan.labels_, idx)
        if len(lab) > 0 and 800 > np.unique(lab).sum() > 1:
            feat = np.delete(features, idx, 0)
            n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            sh_score[0, i, j] = silhouette_score(feat, lab)
            ch_score[0, i, j] = calinski_harabasz_score(feat, lab)
            db_score[0, i, j] = davies_bouldin_score(feat, lab)
            sh_score[1, i, j] = n_clusters_
            ch_score[1, i, j] = n_clusters_
            db_score[1, i, j] = n_clusters_


#X, Y = np.meshgrid(eps, min_samples)
#ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, sh_score)
#ax.plot_surface(X, Y, db_score)
#ax.plot_surface(X, Y, db_score)
#plt.show()
#plt.savefig('assignment-2/IMGs/task6/sh_score.png')

