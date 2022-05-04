import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.distributions.empirical_distribution import ECDF

n_fig = 4
E = 1
N = [2, 3, 4, 5, 6, 7, 8]
colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

cwd = os.getcwd()
features = pd.read_csv(cwd + '/assignment-2/data/data.csv', index_col=0).to_numpy()
labels = pd.read_csv(cwd + '/assignment-2/data/labels.csv').drop(columns='Unnamed: 0')

features_variancefiltered = VarianceThreshold(threshold=2.2343).fit_transform(features)
features_variancefiltered_normalized = StandardScaler().fit_transform(features_variancefiltered)
p = 10
pca = PCA(n_components=p)
features = pca.fit_transform(features_variancefiltered_normalized)

n_points = 50
eps = np.arange(1, n_points+1, 1)
min_samples = np.arange(1, n_points+1, 1)

sh_score = np.zeros((n_points, n_points))
ch_score = np.zeros((n_points, n_points))
db_score = np.ones((n_points, n_points))
sh_params = {}
ch_params = {}
db_params = {}

for i, e in enumerate(eps):
    for j, ms in enumerate(min_samples):
        dbscan = DBSCAN(eps = e, min_samples = ms).fit(features)
        idx = np.where(dbscan.labels_ == -1)
        lab = np.delete(dbscan.labels_, idx)
        feat = np.delete(features, idx, 0)
        if len(lab) > 0 and 800 > np.unique(lab).sum() > 1:
            n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            sh_score[i, j] = silhouette_score(feat, lab)
            ch_score[i, j] = calinski_harabasz_score(feat, lab)
            db_score[i, j] = davies_bouldin_score(feat, lab)
            if 2 <= n_clusters_ <= 10:
                if n_clusters_ not in sh_params or sh_score[i,j] > sh_params[n_clusters_][0]:
                    sh_params[n_clusters_] = (sh_score[i,j], eps[i], min_samples[j])
                if n_clusters_ not in ch_params or ch_score[i,j] > ch_params[n_clusters_][0]:
                    ch_params[n_clusters_] = (ch_score[i,j], eps[i], min_samples[j])
                if n_clusters_ not in db_params or ch_score[i,j] < ch_params[n_clusters_][0]:
                    db_params[n_clusters_] = (db_score[i,j], eps[i], min_samples[j])


X, Y = np.meshgrid(eps, min_samples)
ax = plt.axes(projection='3d')
ax.set_title('DBSCAN silhouette score - Parameter sweep')
ax.set(xlabel='epsilon', ylabel='min_samples')
ax.plot_surface(X, Y, sh_score)
#ax.plot_surface(X, Y, db_score)
#ax.plot_surface(X, Y, db_score)
#plt.show()
print(sh_params)
plt.savefig('assignment-2/IMGs/task6/sh_score.png')
