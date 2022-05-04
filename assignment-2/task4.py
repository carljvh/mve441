import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

n_fig = 4
E = 1
N = [2, 3, 4, 5, 6, 7, 8]
colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

feat = pd.read_csv('assignment-2/data/pc.csv', index_col=0)
lab = pd.read_csv('assignment-2/data/labels.csv').drop(columns='Unnamed: 0')

'''
data = feat.join(lab['Class'])
class_names = lab['Class'].unique()

sh_score = np.zeros((2,len(N)))
ch_score = np.zeros((2,len(N)))
db_score = np.zeros((2,len(N)))

for e in np.arange(E):
    for n in N:
        kmeans = KMeans(n_clusters=n).fit(feat)
        kmeans_pred = kmeans.labels_
        sh_score[0,n-2] += silhouette_score(feat, kmeans_pred)/E
        ch_score[0,n-2] += calinski_harabasz_score(feat, kmeans_pred)/E
        db_score[0,n-2] += davies_bouldin_score(feat, kmeans_pred)/E
        
        agglomerative = AgglomerativeClustering(n_clusters=n).fit(feat)
        agglo_pred = agglomerative.labels_
        sh_score[1,n-2] += silhouette_score(feat, agglo_pred)/E
        ch_score[1,n-2] += calinski_harabasz_score(feat, agglo_pred)/E
        db_score[1,n-2] += davies_bouldin_score(feat, agglo_pred)/E

plt.plot(N, sh_score[0,:], label='K-Means')
plt.plot(N, sh_score[1,:], label='Agglomerative')
plt.legend()
plt.savefig('pics/metrics/sh_score.png')
plt.clf()

plt.plot(N, ch_score[0,:], label='K-Means')
plt.plot(N, ch_score[1,:], label='Agglomerative')
plt.legend()
plt.savefig('pics/metrics/ch_score.png')
plt.clf()

plt.plot(N, db_score[0,:], label='K-Means')
plt.plot(N, db_score[1,:], label='Agglomerative')
plt.legend()
plt.savefig('pics/metrics/db_score.png')
plt.clf()
'''

x = np.linspace(-50, 80, 200)
y = np.linspace(-50, 70, 200)
xx, yy = np.meshgrid(x, y)
'''
X = pd.read_csv('../data/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0)
X = VarianceThreshold(threshold=2.2343).fit_transform(X)
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
pca = pca.fit(X)
X = pca.inverse_transform(np.vstack((xx.ravel(), yy.ravel())).T)
print(X.shape)
print(feat.shape)
'''


kmeans = KMeans(n_clusters=5).fit(feat)

print(np.vstack((xx.ravel(), yy.ravel())).T.shape)
#kmeans_pred = kmeans.predict(X)

#plt.scatter(xx.ravel(), yy.ravel(), c=colors[kmeans_pred])
#plt.savefig('pics/pred_vs_true/test.png')


