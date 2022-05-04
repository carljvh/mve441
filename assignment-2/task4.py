import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from copy import deepcopy

n_fig = 4
E = 1
N = [2, 3, 4, 5, 6, 7, 8]
colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

cwd = os.getcwd()
feat = pd.read_csv(cwd + '\\data\\data.csv', index_col=0)
lab = pd.read_csv(cwd + '\\data\\labels.csv').drop(columns='Unnamed: 0')
#feat = pd.read_csv('assignment-2/data/pc.csv', index_col=0)
#lab = pd.read_csv('assignment-2/data/labels.csv').drop(columns='Unnamed: 0')


data = feat.join(lab['Class'])
class_names = lab['Class'].unique()

sh_score = np.zeros((2,len(N)))
ch_score = np.zeros((2,len(N)))
db_score = np.zeros((2,len(N)))
'''
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
#plt.plot(N, sh_score[0,:], label='K-Means')
#plt.plot(N, sh_score[1,:], label='Agglomerative')
#plt.legend()
#plt.savefig('pics/metrics/sh_score.png')
#plt.clf()
#plt.plot(kmeans_pred)
#plt.show()

# plot true labels
n_PC = 2
pca = PCA(n_components=n_PC)
pc = pca.fit_transform(feat)
pds = pd.DataFrame(pc, columns=['PC1', 'PC2'])
finalDF = pd.concat([pds, lab.reset_index()['Class']], axis=1)
targets = ['PRAD', 'LUAD', 'BRCA', 'COAD', 'KIRC']
colors = ['r', 'g', 'b', 'y', 'c']
grid = sns.pairplot(finalDF, hue="Class", plot_kws={"s": 15})
plt.show()

# plot kmeans
pca = PCA(n_components=n_PC) # n_components ?
principal_components = pca.fit_transform(feat)
principalDF = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
finalDF = pd.concat([principalDF, lab.reset_index()['Class']], axis=1)
cluster_df = deepcopy(finalDF)
labels = cluster_df.pop('Class')
cluster_data = cluster_df.values
kmeans = KMeans(n_clusters=5, init ='k-means++', random_state=0).fit(cluster_data)
kmeans_labels = kmeans.labels_
pairplot_labels = [str(lab) for lab in kmeans_labels]
kmeans_df = finalDF.copy()
kmeans_df = kmeans_df.drop(columns=['Class'])
kmeans_df['Class'] = pairplot_labels
targets = ['PRAD', 'LUAD', 'BRCA', 'COAD', 'KIRC']
colors = ['r', 'g', 'b', 'c', 'y']
grid = sns.pairplot(kmeans_df, hue="Class",  plot_kws={"s": 15})
plt.show()
'''
# plot AgglomerativeClustering
pca = PCA(n_components=2) # n_components ?
principal_components = pca.fit_transform(feat)
principalDF = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
finalDF = pd.concat([principalDF, lab.reset_index()['Class']], axis=1)
cluster_df = deepcopy(finalDF)
labels = cluster_df.pop('Class')
cluster_data = cluster_df.values
agg = AgglomerativeClustering(n_clusters=5).fit(cluster_data)
agg_labels = agg.labels_
pairplot_labels = [str(lab) for lab in agg_labels]
agg_df = finalDF.copy()
agg_df = agg_df.drop(columns=['Class'])
agg_df['Class'] = pairplot_labels
targets = ['PRAD', 'LUAD', 'BRCA', 'COAD', 'KIRC']
colors = ['r', 'g', 'b', 'c', 'y']
grid = sns.pairplot(agg_df, hue="Class",  plot_kws={"s": 15})
plt.show()


nylab = lab.to_numpy()
fails = []
n=0
nyagg = []
for i in range (0,len(agg_labels)):
    if agg_labels[i] == 0 :
        nyagg.append('PRAD')
    if agg_labels[i] == 1 :
        nyagg.append('LUAD')
    if agg_labels[i] == 2 :
        nyagg.append('BRCA')
    if agg_labels[i] == 3 :
        nyagg.append('COAD')
    if agg_labels[i] == 4 :
        nyagg.append('KIRC')
print(np.shape(nyagg))
print(np.shape(nylab))
print(type(nyagg))
print(type(nylab))
nyagg = np.array(nyagg)
print(type(nyagg))
nyagg = np.reshape(nyagg, [(801,1)])
for i in range (0, len (nylab)):
    if nylab[i] == nyagg[i]:
        a=5
    else:
        fails.append(n)
        n += 1
print(np.shape(nyagg))
print(np.shape(nylab))
print(fails)

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()
'''
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


#print(np.vstack((xx.ravel(), yy.ravel())).T.shape)
#kmeans_pred = kmeans.predict(X)

#plt.scatter(xx.ravel(), yy.ravel())c=colors[kmeans_pred])
#plt.savefig('pics/pred_vs_true/test.png')


