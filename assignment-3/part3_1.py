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
    return lasso_clf.alphas_[idx_alpha]


# returns dataframe containing feature frequency and average coefficient values
def generate_stability_stats(data, labels, alpha, M=1, sample_prop=0.95):
    df = data.join(labels['Class'])
    n_features = len(data.columns)
    F = np.zeros(n_features)
    I = np.zeros_like(F)
    for m in range(M):
        sample_data = df.sample(frac=sample_prop, replace=True)
        lasso = Lasso(alpha=alpha).fit(sample_data.iloc[:,0:n_features-1], sample_data['Class'])
        for j in range(n_features):
            if abs(lasso.coef_[j]) > 0:
                F[j] += 1
        I += lasso.coef_

    #F = F / M
    I = I / M
    # pd.DataFrame({'frequency': F, 'coef_abs': np.absolute(I)})
    return pd.DataFrame({'frequency': F})


def generate_alpha(alpha_type: str, X_train, y_train, n_folds):
    lasso_cv = LassoCV(cv=n_folds, max_iter=2000).fit(X_train,y_train)
    if alpha_type == 'max':
        return lasso_cv.alpha_
    idx = get_lambda_1se_idx(lasso_cv, n_folds)
    return lasso_cv.alphas_[idx]


def generate_alphas(X_train, y_train, n_folds):
    lasso_cv = LassoCV(cv=n_folds, max_iter=2000).fit(X_train,y_train)
    std = np.std(lasso_cv.alphas_)
    return [alpha for alpha in lasso_cv.alphas_ if (lasso_cv.alpha_ - std/n_folds) <= alpha <= (lasso_cv.alpha_ + std/n_folds)]


"""
def analyze_stability(param_name, n_test=1000, p=1000, beta_scale=5.0, n_folds=5, **kwargs):
    top_n = 1000
    top_data = []
    if param_name == 'n_samples':
        for value in kwargs[param_name]:
            X, y, beta = simulate_data(n=value+n_test, p=p, rng=np.random.default_rng(seed=42), sparsity=kwargs['sparsity'], SNR=kwargs['SNR'], beta_scale=beta_scale)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(value+n_test), random_state=42)
            alpha = generate_alpha('max', X_train, y_train, n_folds)
            df = generate_stability_stats(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha)
            top_data.append(df)
            #top_data.append((df.nlargest(top_n, ['frequency', 'coef_abs'])))

    if param_name == 'sparsity':
        for value in kwargs[param_name]:
            X, y, beta = simulate_data(n=kwargs['n_samples']+n_test, p=p, rng=np.random.default_rng(seed=42), sparsity=value, SNR=kwargs['SNR'], beta_scale=beta_scale)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(kwargs['n_samples']+n_test), random_state=42)
            alpha = generate_alpha('max', X_train, y_train, n_folds)
            df = generate_stability_stats(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha)
            top_data.append(df)
            #top_data.append((df.nlargest(top_n, ['frequency', 'coef_abs'])))

    if param_name == 'SNR':
        for value in kwargs[param_name]:
            X, y, beta = simulate_data(n=kwargs['n_samples']+n_test, p=p, rng=np.random.default_rng(seed=42), sparsity=kwargs['sparsity'], SNR=value, beta_scale=beta_scale)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(kwargs['n_samples']+n_test), random_state=42)
            alpha = generate_alpha('max', X_train, y_train, n_folds)
            df = generate_stability_stats(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha)
            top_data.append(df)
            #top_data.append((df.nlargest(top_n, ['frequency', 'coef_abs'])))
            
    if param_name == 'alpha':
        X, y, beta = simulate_data(n=kwargs['n_samples']+n_test, p=p,rng=np.random.default_rng(seed=42), sparsity=kwargs['SNR'], SNR=kwargs['SNR'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(kwargs['n_samples']+n_test), random_state=42)
        alphas = generate_alphas(X_train, y_train, n_folds)
        for alpha in alphas:
            df = generate_stability_stats(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha)
            top_data.append(df)
            #top_data.append((df.nlargest(top_n, ['frequency', 'coef_abs'])))
    return top_data    


kwargs_SNR = {'SNR': SNR,'n_samples': n_samples[2], 'sparsity': sparsity[3]}
kwargs_n_samples = {'SNR': SNR[2],'n_samples': n_samples, 'sparsity': sparsity[3]}
kwargs_sparsity = {'SNR': SNR[2],'n_samples': n_samples[2], 'sparsity': sparsity}
kwargs_alpha = {'SNR': SNR[2],'n_samples': n_samples[2], 'sparsity': sparsity[3]}


stab_data = analyze_stability(param_name='SNR', **kwargs_SNR)
plt.bar(x=stab_data[2].index.tolist(), height=stab_data[2]['frequency'])
plt.show()
"""

# fix average over iterations
# fix 
# fix number of lambdas (too many)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test/(n+n_test), random_state=42)
#alphas = generate_alphas(X, y, n_folds)
M = 50
n_folds = 5
sample_prop = 0.95

SNR = np.array([0.1, 1.0, 2.0])
n_samples = np.array([40, 100, 150])
n_features = 200
sparsity = np.array([0.75, 0.9, 0.95, 0.99])

X, y, beta = simulate_data(n=n_samples[2], p=n_features, rng=np.random.default_rng(seed=42), sparsity=sparsity[3], SNR=SNR[0])
counts_min = np.zeros(n_features)
counts_1se = np.zeros_like(counts_min)
for m in range(M):
    print("bootstrap sample: %s" % m)
    df = pd.DataFrame(X)
    df['y'] = y
    bs_sample = df.sample(frac=sample_prop, replace=True)
    X_bs = bs_sample.iloc[:,0:n_features].values
    y_bs = bs_sample.loc[:, 'y'].to_numpy()

    lasso_min = LassoCV(cv=n_folds, max_iter = 3000).fit(X_bs, y_bs)
    alpha_min = lasso_min.alphas_

    alpha_1se = get_lambda_1se_idx(lasso_min, n_folds)
    lasso_1se = Lasso(alpha=alpha_1se).fit(X_bs, y_bs)
    for j in range(n_features):
        if abs(lasso_min.coef_[j]) > 0:
            counts_min[j] += 1
        if abs(lasso_1se.coef_[j]) > 0:
            counts_1se[j] += 1

plt.bar(x=np.linspace(1,n_features,n_features), height=counts_min, width=0.5)
plt.show()

"""
for alpha in alphas:
    df = generate_stability_stats(pd.DataFrame(X_test), pd.DataFrame({'Class': y_test}), alpha)
    top_n = 1000
    top_data = df.nlargest(top_n, ['frequency', 'coef_abs'])
        plt.bar(x=top_data.index.tolist(), height=top_data['frequency'])
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
"""