import pandas as pd
import matplotlib as mt
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, recall_score
from texttable import Texttable
from numpy import random
import matplotlib.pyplot as plt

def mislabel_data(df, percentage):
    for i in range(len(df.columns)):
        r = random.rand()
        if r < percentage:
            df.iat[i, 30] = 1
    return df

# Load UCI breast cancer dataset with column names and remove ID column
uci_bc_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    sep=",",
    header=None,
    names=[
        "id_number", "diagnosis", "radius_mean",
        "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean",
        "concavity_mean","concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se",
        "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ],).drop("id_number", axis=1)

y = uci_bc_data.diagnosis.map({"B": 0, "M": 1})
X = uci_bc_data.drop("diagnosis", axis=1)

df = pd.concat([X,y], axis=1, ignore_index = True)
real_p = 0.10
data = mislabel_data(df, real_p)

votes = {}
runs = 10
for i in range(runs):
    partition = data.sample(frac = 1/runs)
    X = partition.iloc[:, 0:29]
    y = partition.iloc[:,[30]]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train.to_numpy().ravel())
    y_pred = clf.predict(X_test)

    index_list = list(y_test.index.values)
    for i, row_idx in enumerate(index_list):
        if y_pred[i] == y_test.iat[i,0]:
            if row_idx not in votes:
                votes[row_idx] = (1,1)
            else:
                votes[row_idx] = tuple(map(lambda i, j: i + j, votes[row_idx], (1,1)))
        else:
            if row_idx not in votes:
                votes[row_idx] = (0,1)
            else:
                votes[row_idx] = tuple(map(lambda i, j: i + j, votes[row_idx], (0,1)))


vote_limit = 0.5
counter = 0
for key,value in votes.items():
    correct_votes = value[0]
    total_votes = value[1]
    if correct_votes/total_votes < vote_limit:
        counter += 1

found_p = counter/len(df.index)
print("%f percent" % (found_p/real_p*100))