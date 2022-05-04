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

cwd = os.getcwd()
features = pd.read_csv(cwd + '/assignment-2/data/data.csv', index_col=0).to_numpy()
labels = pd.read_csv(cwd + '/assignment-2/data/labels.csv').drop(columns='Unnamed: 0')

features_variancefiltered = VarianceThreshold(threshold=2.2343).fit_transform(features)
features_variancefiltered_normalized = StandardScaler().fit_transform(features_variancefiltered)
p = 10
pca = PCA(n_components=p)
features = pca.fit_transform(features_variancefiltered_normalized)

# Plot while tweaking hyperparameters to find best fits for k = 2,...,5
params = {2: (45,115), 3: (43,80), 4: (40,70), 5: (35,58)}
# Plot result
fig, axs = plt.subplots(2, 2)
fig.suptitle('DBSCAN for different hyperparameter values')
for i, ax in enumerate(axs.flatten()):
    eps = params[i+2][0]
    ms = params[i+2][1]
    db = DBSCAN(eps=eps, min_samples=ms).fit(features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        
        class_member_mask = (labels == k)
        xy = features[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        markeredgecolor='k', markersize=14)
        xy = features[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        markeredgecolor='k', markersize=6)
        ax.set_title("Estimated no. clusters: %d for epsilon = %s, min_samples = %s" % (n_clusters_, str(eps), str(ms)))
        ax.set(xlabel='PC1', ylabel='PC2')
plt.show()
plt.savefig('assignment-2/IMGs/task6/dbscan_clusters.png')

k_values = list(params.keys())
kmin = np.min(k_values)
kmax = np.max(k_values) + 1
K = range(kmin, kmax)
n = len(features[:,0])
n_subsets = 15
M = np.zeros((n,n))
J = np.zeros((n,n))
C = []

for k in K:
    M = np.zeros((n, n))
    J = np.zeros((n, n))
    for m in np.arange(n_subsets):
        subX = pd.DataFrame(data=features).sample(n=int(0.8 * n))
        eps = params[k][0]
        ms = params[k][1]
        dbscan = DBSCAN(eps=eps, min_samples=ms).fit(subX) 
        pred = dbscan.labels_
        for i, c1 in enumerate(pred):
            for j, c2 in enumerate(pred):
                if c1 == c2:
                    M[subX.index[i], subX.index[j]] += 1
                J[subX.index[i], subX.index[j]] += 1
    C.append(np.divide(M, J))

q1 = 0.01
q2 = 0.99
pac_scores = np.zeros(kmax-kmin)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle("DBSCAN applied to (HiSeq) PANCAN - %d features" % p)
ax1.set(title="eCDF vs. number of clusters", xlabel="Concensus matrix values", ylabel="Probability")
for i, c in enumerate(C): 
    ecdf = ECDF(c.flatten())
    ax1.plot(ecdf.x, ecdf.y, label="k-value: %s" % str(kmin+i))
    pac_scores[i] = ecdf(q2)-ecdf(q1)

ax1.legend(loc=0)
ax2.set(title="PAC-score vs. number of clusters", xlabel="k-value", ylabel="PAC-score")
ax2.plot(np.linspace(kmin, kmax-1,kmax-kmin), pac_scores, '--bo')
plt.savefig('assignment-2/IMGs/task6/dbscan_ecdf.png')