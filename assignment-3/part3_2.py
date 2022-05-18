import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegressionCV

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


M = 50
n_folds = 5
sample_prop = 0.95

SNR = 2
n_samples = np.array([500, 100, 150])
n_features = 200
sparsity = 0.9

X, y, beta = simulate_data(n=n_samples[1], p=n_features, rng=np.random.default_rng(seed=42), sparsity=sparsity, SNR=SNR)

df = pd.DataFrame(X)
df['y'] = y

coefMatrix = [np.zeros(n_features),np.zeros(n_features),np.zeros(n_features) ]

def makeBinary(a):
    a[a!=0] = 1
    return a.astype("int32")


B = 50
for iter in range(B):
    print(iter)
    # Sampling
    bs_sample = df.sample(frac=sample_prop, replace=True)
    X_bs = bs_sample.iloc[:,0:n_features].values
    y_bs = bs_sample.loc[:, 'y'].to_numpy()

    # Lasso Cross Validation
    lasso_min = LassoCV(cv=n_folds, max_iter = 3000).fit(X_bs, y_bs)
    alpha_min = lasso_min.alpha_
    alpha_1se = get_lambda_1se_idx(lasso_min, n_folds)
    alphas = [0.8*alpha_1se, alpha_1se, 1.2*alpha_1se]
    index=0
    for alph in alphas:
        lassoModel = Lasso(alpha=alph).fit(X_bs, y_bs)

        lassol1s = lassoModel.coef_
        bin = makeBinary(lassol1s)
        coefMatrix[index] += bin
        index +=1

plt.bar(x=np.linspace(1,n_features,n_features), height=coefMatrix[1], width=0.5)
plt.show()
thresholds = np.array([0.7, 0.75, 0.8, 0.85, 0.9])

for al in range(0, len(alphas)):
    sensitivityarray = []
    specificityyarray = []
    for th in range(0,len(thresholds)):
        threshold = thresholds[th]
        threshold_counts_min = []
        threshold_index = []
        for i in range(0,len(coefMatrix[1])):
            if coefMatrix[al][i] > threshold * B:
                threshold_counts_min.append(coefMatrix[al][i])
                threshold_index.append(i)
        ypred = np.zeros(200)
        for i in range(0,len(threshold_index)):
            ypred[threshold_index[i]] = 1
        betabin = makeBinary(beta)
        tn, fp, fn, tp = confusion_matrix(betabin,ypred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivityarray.append(sensitivity)
        specificityyarray.append(specificity)
    print(sensitivityarray)
    plt.plot(thresholds, sensitivityarray, label = alphas[al])
    plt.ylim(ymin=0)
plt.legend()
plt.show()

# get lambdas
# LassoCV -> lmin lse 
# fix lmin multipliers list of 3
# For each lambdaMult 
# Lasso shit
# få ut listor med hur många gpnger feats komemr på 50 iters
# Kör thresh hold 60 70 80 90
#