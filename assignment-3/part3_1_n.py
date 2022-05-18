import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

def simulate_data(n, p, rng, *, sparsity=0.95, SNR=2.0, beta_scale=5.0, sd=1):
    """Simulate data for Project 3, Part 1.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    rng : numpy.random.Generator
        Random number generator (e.g. from numpy.random.default_rng)
    sparsity : float in (0, 1)
        Percentage of zero elements in simulated regression coefficients
    SNR : positive float
        Signal-to-noise ratio (see explanation above)
    beta_scale : float
        Scaling for the coefficient to make sure they are large
    sd : float
        lower values gives more correlations

    Returns
    -------
    X : n x p numpy.array
        Matrix of features
    y : n numpy.array
        Vector of responses
    beta : p numpy.array
        Vector of regression coefficients
    """
    x = rng.standard_normal((n, 1))
    X = np.tile(x, (1, p)) + rng.normal(loc=0, scale=sd, size=(n, p))
    
    q = int(np.ceil((1.0 - sparsity) * p))
    beta = np.zeros((p,), dtype=float)
    beta[:q] = beta_scale * rng.standard_normal(size=(q,))
    
    sigma = np.sqrt(np.sum(np.square(X @ beta)) / (n - 1)) / SNR

    y = X @ beta + sigma * rng.standard_normal(size=(n,))

    # Shuffle columns so that non-zero features appear
    # not simply in the first (1 - sparsity) * p columns
    idx_col = rng.permutation(p)
    
    return X[:, idx_col], y, beta[idx_col]


def get_lambda_1se_idx(lasso_clf, n_folds):
    cv_mean = np.mean(lasso_clf.mse_path_, axis=1)
    cv_std = np.std(lasso_clf.mse_path_, axis=1)
    idx_min_mean = np.argmin(cv_mean)
    idx_alpha = np.where(
        (cv_mean <= cv_mean[idx_min_mean] + cv_std[idx_min_mean] / np.sqrt(n_folds)) &
        (cv_mean >= cv_mean[idx_min_mean])
        )[0][0]
    return lasso_clf.alphas_[idx_alpha]


def calc_TPR(betas, counts):
    betas[np.absolute(betas) > 0] = 1
    counts[counts < 0.9*M] = 0
    counts[counts >= 0.9*M] = 1
    TPR = []
    for i in range(len(betas)):
        TN, FP, FN, TP = confusion_matrix(betas[i], counts[i]).ravel()
        tpr = TP/(TP+FN)
        TPR.append(np.around(tpr, decimals=2))
    return TPR


M = 50
n_folds = 5
sample_prop = 0.95

SNR = np.array([0.2, 1.0, 5.0])
n_samples = np.array([40, 100, 150])
n_features = 200
sparsity = np.array([0.75, 0.9, 0.95, 0.99])

counts = np.zeros((len(n_samples),n_features))
betas = np.zeros_like(counts)
for i, n in enumerate(n_samples):
    X, y, beta = simulate_data(n=n, p=n_features, rng=np.random.default_rng(seed=42), sparsity=sparsity[1], SNR=SNR[2])
    df = pd.DataFrame(X)
    df['y'] = y
    betas[i] = beta
    
    for m in range(M):
        print("bootstrap sample #%s" % str(m+1))
        bs_sample = df.sample(frac=sample_prop, replace=True)
        X_bs = bs_sample.iloc[:,0:n_features].values
        y_bs = bs_sample.loc[:, 'y'].to_numpy()

        lasso_min = LassoCV(cv=n_folds, max_iter = 3000).fit(X_bs, y_bs)
        alpha_1se = get_lambda_1se_idx(lasso_min, n_folds)
        lasso_1se = Lasso(alpha=alpha_1se).fit(X_bs, y_bs)

        for j in range(n_features):
            if abs(lasso_1se.coef_[j]) > 0:
                counts[i][j] += 1

TPR = calc_TPR(betas.copy(), counts.copy())

fig, axes = plt.subplots(1, 3)
fig.suptitle('Counts for features with SNR=%s, p=%s, s=%s varied over n' % (SNR[2], n_features, sparsity[1]))
for i, ax in enumerate(axes.flatten()):
    df = pd.DataFrame({'counts': counts[i][:]})
    df = df[df.counts >= 0.9*M]
    indices = np.arange(len(df.index.values))
    labels = df.index.values
    ax.bar(x=indices, height=df['counts'], width=0.5, tick_label=labels)
    ax.set_xticklabels(labels, rotation=70)
    ax.set(title=("n=%s, TPR=%s" % (n_samples[i], TPR[i])), xlabel="features", ylabel="counts")

plt.savefig("assignment-3/images/n_samples_90.png")
plt.show()