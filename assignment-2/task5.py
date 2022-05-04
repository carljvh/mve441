from cProfile import label
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.distributions.empirical_distribution import ECDF

cwd = os.getcwd()
labels = pd.read_csv(cwd + '/assignment-2/data/labels.csv').drop(columns='Unnamed: 0')
features = pd.read_csv(cwd + '/assignment-2/data/data.csv', index_col=0)

features_variancefiltered = VarianceThreshold(threshold=2.2343).fit_transform(features)
features_variancefiltered_normalized = StandardScaler().fit_transform(features_variancefiltered)
p = 521
pca = PCA(n_components=p)
principal_components = pca.fit_transform(features_variancefiltered_normalized)
X = pd.DataFrame(data = principal_components)

kmin = 2
kmax = 11
K = range(kmin, kmax)
n = len(X.index)
n_subsets = 15
M = np.zeros((n,n))
J = np.zeros((n,n))
C = []

for k in K:
        M = np.zeros((n, n))
        J = np.zeros((n, n))
        for m in np.arange(n_subsets):
            subX =  X.sample(n=int(0.8 * n))
            kmeans = KMeans(n_clusters=k).fit(subX) 
            #subX['labels'] = kmeans.labels_
            pred = kmeans.labels_
            for i, c1 in enumerate(pred):
                for j, c2 in enumerate(pred):
                    if c1 == c2:
                        M[subX.index[i], subX.index[j]] += 1
                    J[subX.index[i], subX.index[j]] += 1
        C.append(np.divide(M, J))

q1 = 0.02
q2 = 0.99
pac_scores = np.zeros(kmax-kmin)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle("k-means applied to (HiSeq) PANCAN - %d features" % p)
ax1.set(title="eCDF vs. number of clusters", xlabel="Concencus matrix values", ylabel="Probability")
for i, c in enumerate(C): 
    ecdf = ECDF(c.flatten())
    ax1.plot(ecdf.x, ecdf.y, label="k-value: %s" % str(kmin+i))
    pac_scores[i] = ecdf(q2)-ecdf(q1)

ax1.legend(loc=0)
ax2.set(title="PAC-score vs. number of clusters", xlabel="k-values", ylabel="PAC-score")
ax2.plot(np.linspace(kmin, kmax-1,kmax-kmin), pac_scores, '--bo')
plt.show()