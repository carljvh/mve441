import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def simulate_data(n, p, rng, *, sparsity=0.95, SNR=2.0, beta_scale=5.0):
    """Simulate data for Project 3, Part 1.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    rng : numpy.random.Generator
        Random number generator (e.g. from `numpy.random.default_rng`)
    sparsity : float in (0, 1)
        Percentage of zero elements in simulated regression coefficients
    SNR : positive float
        Signal-to-noise ratio (see explanation above)
    beta_scale : float
        Scaling for the coefficient to make sure they are large

    Returns
    -------
    X : `n x p` numpy.array
        Matrix of features
    y : `n` numpy.array
        Vector of responses
    beta : `p` numpy.array
        Vector of regression coefficients
    """
    X = rng.standard_normal(size=(n, p))
    
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
    return idx_alpha


iter = 1
n_test = 1000
n_datapoints = np.array([200, 500, 750])
sparsity = np.array([0.75, 0.9, 0.95, 0.99])
n_folds = 5
alpha_min = np.zeros((len(n_datapoints), len(sparsity)), dtype=float)
alpha_1se = np.zeros_like(alpha_min)
mse_min_train = np.zeros_like(alpha_min)
mse_min_test = np.zeros_like(alpha_min)
mse_1se_train = np.zeros_like(alpha_min)
mse_1se_test = np.zeros_like(alpha_min)
for it in range(iter):
    for i,n in enumerate(n_datapoints):
        for j,s in enumerate(sparsity):
            X, y, beta = simulate_data(n=n+n_test, p=1000,rng=np.random.default_rng(), sparsity=s)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(n+n_test), random_state=42)
            lasso_cv = LassoCV(cv=n_folds).fit(X_train,y_train)

            alpha_min_ij = lasso_cv.alpha_
            alpha_min[i,j] += alpha_min_ij
            idx_alpha_1se = get_lambda_1se_idx(lasso_cv, n_folds)
            alpha_1se_ij = lasso_cv.alphas_[idx_alpha_1se]
            alpha_1se[i,j] += alpha_1se_ij

            mse_min_train[i,j] += np.min(np.mean(lasso_cv.mse_path_, axis=1))
            mse_1se_train[i,j] += np.mean(lasso_cv.mse_path_, axis=1)[idx_alpha_1se]

            lasso_min = Lasso(alpha=alpha_min_ij).fit(X_test,y_test)
            y_pred = lasso_min.predict(X_test)
            mse_min_test[i,j] += mean_squared_error(y_test, y_pred)
            
            lasso_1se = Lasso(alpha=alpha_1se_ij).fit(X_test,y_test)
            y_pred = lasso_1se.predict(X_test)
            mse_1se_test[i,j] += mean_squared_error(y_test, y_pred)

alpha_min = alpha_min / iter
alpha_1se = alpha_1se / iter
mse_min_train = mse_min_train / iter
mse_min_test = mse_min_test / iter
mse_1se_train = mse_1se_train / iter
mse_1se_test = mse_1se_test / iter

