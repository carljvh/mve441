import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    return idx_alpha


# returns feature frequency matrix and feature average coefficient value matrix
def generate_confidence_data(data, labels, alpha, M=1, sample_prop=0.95):
    df = data.join(labels['Class'])
    n_features = len(data.columns)
    F = np.zeros(n_features)
    I = np.zeros_like(F)
    for m in range(M):
        sample_data = df.sample(frac=sample_prop, replace=True)
        lasso = Lasso(alpha=alpha).fit(sample_data.iloc[:,0:200], sample_data['Class'])
        for j in range(n_features):
            if abs(lasso.coef_[j]) > 0:
                F[j] += 1
        I += lasso.coef_
    F = F / M
    I = I / M
    return F, I


# returns a list with a dataframe for each class containing top n features, rated by frequency and abs(size) of coefficient
def select_top_n_genes(top_n, freq_array, coef_avg_array):
    df = pd.DataFrame({'frequency': freq_array, 'coef_abs': np.absolute(coef_avg_array)})
    return df.nlargest(top_n, ['frequency', 'coef_abs'])


iter = 5
n_test = 1000
n_datapoints = np.array([200, 500, 750])
sparsity = np.array([0.75, 0.9, 0.95, 0.99])
n_folds = 5

for it in range(iter):
    for i,n in enumerate(n_datapoints):
        for j,s in enumerate(sparsity):
            X, y, beta = simulate_data(n=n+n_test, p=1000,rng=np.random.default_rng(), sparsity=s, SNR=2.0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(n+n_test), random_state=42)
            lasso_cv = LassoCV(cv=n_folds).fit(X_train,y_train)
            std = np.std(lasso_cv.alphas_)
            alphas = [alpha for alpha in lasso_cv.alphas_ if (lasso_cv.alpha_ - std/n_folds) <= alpha <= (lasso_cv.alpha_ + std/n_folds)]

            for alpha in alphas:
                F, I = generate_confidence_data(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha, M=1)
                top_n = 5
                top_data = select_top_n_genes(top_n=top_n, freq_array=F, coef_avg_array=I)
                top_data.plot.bar(y='frequency', use_index=True, rot=0, ax=ax)
                plt.show()
                exit(0)
                fig, axes = plt.subplots(2, 3)
                fig.suptitle('Frequencies for n_samples = %s, sparsity=%s, alpha=%s ' % (n_datapoints, sparsity, alpha))
                axes[1][2].set_visible(False)
                for i, ax in enumerate(axes.flatten()):
                    if i == 5:
                        break
                    top_data[i].plot.bar(y='frequency', use_index=True, rot=0, ax=ax)
                plt.show()