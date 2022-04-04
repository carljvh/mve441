from sklearn import tree
import numpy as np
import pandas as pd
import random
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


n = 100
runs = 10
result = evaluate(n, runs)
print("mean accuracy: %f (std %f)" % (result[0],result[1]))
