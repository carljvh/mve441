from sklearn import tree
import numpy as np
import pandas as pd
import random
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_uni_data(n, x1_lim, x2_lim):
    X = np.zeros([n,2])
    y = np.zeros(n)

    for i in range(n):
            X[i,0] = random.uniform(0,10)
            X[i,1] = random.uniform(0,10)
            if X[i,0] >=x1_lim and X[i,1] <=x2_lim:
                y[i] = 1
        #  else:  #Not needed
        #     y[i] = 0
    return X, y

    # mu, sigma vectors with length of classes
def generate_gaussian_data(n, mu, sigma):
    X = np.random.normal(mu, sigma, [n,2])
    y = np.zeros(n)
    y[int(n/2):] = 1
    return X, y

 # each class should have individual covar matrix
def generate_mvn_data(n, mu, covar):
    #X = np.random.multivariate_normal(mu, covar, [n,2])
    X = np.random.multivariate_normal(mean=mu[0], cov=covar[0], size=int(n/2))
    X = np.append(X,np.random.multivariate_normal(mean=mu[1], cov=covar[1], size=int(n/2)),axis=0)
    y = np.zeros(n)
    y[int(n/2):] = 1
    return X, y



def evaluate(n_datapoints: int, runs: int):
    accuracy = np.zeros(runs)
    for i in range(runs):
        data = generate_uni_data(n, 5, 4)
        X = data[0]
        y = data[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        cart = tree.DecisionTreeClassifier()
        cart = cart.fit(X_train, y_train)

        y_pred = cart.predict(X_test)
        accuracy[i] = accuracy_score(y_test, y_pred)
    
    return np.mean(accuracy), np.std(accuracy)

def evaluateQDA(n_datapoints: int, runs: int, mean, covar):
    accuracy = np.zeros(runs)
    loss = np.zeros(runs)
    for i in range(runs):
        data = generate_mvn_data(n, mean, covar)
        #data = generate_gaussian_data(n,5,4)
        #print(data)

        X = data[0]
        y = data[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        qda = QuadraticDiscriminantAnalysis()
        qdaFit = qda.fit(X_train,y_train)

        y_pred = qdaFit.predict(X_test)
        accuracy[i] = accuracy_score(y_test, y_pred)

        y_g_train_pred = qda.predict(X_train)
        loss[i] = zero_one_loss(y_train, y_g_train_pred)
       
    
    return np.mean(accuracy), np.std(accuracy), np.mean(loss) 



mu = [[2,2], [5,5]]
covar = [[[1,2],[2,1]],  [[2,1],[1,2]]]

n = 100
runs = 10



result = evaluateQDA(n, runs, mu, covar)
print("mean accuracy: %f (std %f)" % (result[0],result[1]))
print(f"training Loss = {result[2]}")
