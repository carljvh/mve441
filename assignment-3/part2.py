import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegressionCV

def calc_C_1se(clf):
    C_1se = []
    for c in clf.scores_:
        cv_mean = np.mean(clf.scores_[c], axis=0)
        cv_std = np.std(clf.scores_[c], axis=0)
        idx_max_mean = np.argmax(cv_mean)
        n_folds = clf.get_params()['cv']
        idx_C_1se = np.where(
            (cv_mean >= cv_mean[idx_max_mean] - cv_std[idx_max_mean] / np.sqrt(n_folds)) &
            (cv_mean <= cv_mean[idx_max_mean])
        )[0][0]
        C_1se.append(clf.Cs_[idx_C_1se])
    return C_1se


# returns feature frequency matrix and feature average coefficient value matrix
def generate_confidence_data(data, labels, M=50, sample_prop=0.95):
    df = data.join(labels['Class'])
    n_classes = labels['Class'].nunique()
    n_features = len(data.columns)
    F = np.zeros((n_classes, n_features))
    I = np.zeros_like(F)
    for m in range(M):
        sample_data = df.sample(frac=sample_prop, replace=True)
        clf = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", intercept_scaling=10000, multi_class="ovr", scoring="f1", Cs=np.logspace(-4, 4, 30)).fit(sample_data.iloc[:,0:200], sample_data['Class'])
        for i in range(n_classes):
            for j in range(n_features):
                if abs(clf.coef_[i,j]) > 0:
                    F[i,j] += 1
        I += clf.coef_
    F = F / M
    I = I / M
    return F, I


# returns a list with a dataframe for each class containing top n features, rated by frequency and abs(size) of coefficient
def select_top_n_genes(top_n, n_classes, freq_matrix, coef_avg_matrix):
    top_data = []
    F = pd.DataFrame(freq_matrix.T)
    I = pd.DataFrame(coef_avg_matrix.T)
    for i in range(n_classes):
        df = pd.DataFrame({'frequency': F.iloc[:,i].tolist(), 'coef_abs': np.absolute(I.iloc[:,i].tolist())})
        top_data.append(df.nlargest(top_n, ['frequency', 'coef_abs']))
    return top_data


# Task 1
data_folder = os.getcwd() + '/assignment-3/data/'
data_full = pd.read_csv(data_folder + 'data.csv', index_col=0)
labels = pd.read_csv(data_folder + 'labels.csv')
data = pd.DataFrame(SelectKBest(score_func=f_classif, k=200).fit_transform(data_full, labels['Class']))

# Task 2
clf = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", intercept_scaling=10000, multi_class="ovr", scoring="f1", Cs=np.logspace(-4, 4, 30)).fit(data, labels['Class'])
C_max = clf.C_
C_1se = calc_C_1se(clf)
print("C_max: %s" % C_max)
print("C_1se: %s" % C_1se)

# Task 3
F, I = generate_confidence_data(data, labels)
top_n = 10
top_data = select_top_n_genes(top_n=top_n, n_classes=labels['Class'].nunique(), freq_matrix=F, coef_avg_matrix=I)

fig, axes = plt.subplots(2, 3)
fig.suptitle('Frequencies for the top %s selected genes for each diagnosis' % top_n)
axes[1][2].set_visible(False)
for i, ax in enumerate(axes.flatten()):
    if i == 5:
        break
    top_data[i].plot.bar(y='frequency', use_index=True, rot=0, ax=ax)
plt.show()
