from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_uni_data(n_datapoints, x1_lim, x2_lim):
    data = np.zeros([n_datapoints,3])

    for i in range(n_datapoints):
        data[i,0] = random.uniform(0,10)
        data[i,1] = random.uniform(0,10)
        if data[i,0] >= x1_lim and data[i,1] <= x2_lim:
            data[i,2] = 1
    
    return pd.DataFrame(data, columns=["x_1", "x_2", "class"])


 # each class should have individual covar matrix
def generate_mvn_data(n_datapoints, mu, covar):
    X = np.random.multivariate_normal(mean=mu[0], cov=covar[0], size=int(n_datapoints/2))
    X = np.append(X,np.random.multivariate_normal(mean=mu[1], cov=covar[1], size=int(n_datapoints/2)),axis=0)
    y = np.zeros([n_datapoints,1])
    y[int(n_datapoints/2):] = 1
    data = np.append(X, y, axis=1)
    return pd.DataFrame(data, columns=["x_1", "x_2", "class"])


def evaluate(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(dataframe[['x_1', 'x_2']], dataframe['class'], test_size=0.33, random_state=0)

    cart = tree.DecisionTreeClassifier()
    cart = cart.fit(X_train, y_train)
    qda = QuadraticDiscriminantAnalysis()
    qdaFit = qda.fit(X_train,y_train)

    y_pred_cart = cart.predict(X_test)
    y_pred_qda = qdaFit.predict(X_test)

    accuracy_cart = accuracy_score(y_test, y_pred_cart)
    accuracy_qda = accuracy_score(y_test, y_pred_qda)
    
    return accuracy_cart, accuracy_qda


def generate_average_results(n_datapoints, runs, x1_lim, x2_lim, mu, cov):
    accuracy = np.zeros([runs, 4])

    for i in range(runs):
        df_uni = generate_uni_data(n_datapoints, x1_lim, x2_lim)
        result_uni = evaluate(df_uni)
        accuracy[i,0] = result_uni[0] 
        accuracy[i,1] = result_uni[1]

        df_mvn = generate_mvn_data(n_datapoints, mu, cov)
        result_mvn = evaluate(df_mvn)
        accuracy[i,2] = result_mvn[0] 
        accuracy[i,3] = result_mvn[1]
    
    return {'cart': {'uni': (np.mean(accuracy[:,0]), np.std(accuracy[:,0])), 'mvn': (np.mean(accuracy[:,2]), np.std(accuracy[:,2]))}, 
    'qda':{'uni': (np.mean(accuracy[:,1]), np.std(accuracy[:,1])), 'mvn': (np.mean(accuracy[:,3]), np.std(accuracy[:,3]))}}


# Taken from public course material for dit866 on Chalmers 
# (http://www.cse.chalmers.se/~richajo/dit866/backup_2019/lectures/l3/Plotting%20decision%20boundaries.html)
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


# fix more complex data uni
# measure test error via confusion matrices?
n = 500
runs = 20

x1_lim = 5
x2_lim = 4
mu = [[7,2], [2,7]]
cov = [[[1,0],[0,1]],  [[2,1],[1,2]]]

df = generate_uni_data(n, x1_lim, x2_lim)
X_train, X_test, y_train, y_test = train_test_split(df[['x_1', 'x_2']], df['class'], test_size=0.33, random_state=0)
cart = tree.DecisionTreeClassifier()
cart = cart.fit(X_train.to_numpy(), y_train.to_numpy())
plot_decision_boundary(cart, X_test.to_numpy(), y_test.to_numpy())

df = generate_mvn_data(500, mu, cov)
X_train, X_test, y_train, y_test = train_test_split(df[['x_1', 'x_2']], df['class'], test_size=0.33, random_state=0)
qda = QuadraticDiscriminantAnalysis()
qdaFit = qda.fit(X_train.to_numpy(),y_train.to_numpy())
plot_decision_boundary(qdaFit, X_test.to_numpy(), y_test.to_numpy())

result = generate_average_results(n, runs, x1_lim, x2_lim, mu, cov)
print("Uniform data: \nmean accuracy for CART: %f (std %f)" % (result['cart']['uni'][0],result['cart']['uni'][1]))
print("mean accuracy for QDA: %f (std %f) \n" % (result['qda']['uni'][0],result['qda']['uni'][1]))

print("Normal data: \nmean accuracy for CART: %f (std %f)" % (result['cart']['mvn'][0],result['cart']['uni'][1]))
print("mean accuracy for QDA: %f (std %f) \n" % (result['qda']['mvn'][0],result['qda']['mvn'][1]))
