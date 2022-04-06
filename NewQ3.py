# Code from question 3
import pandas as pd
import matplotlib as mt
import numpy as np
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

y = uci_bc_data.diagnosis.map({"B": 0, "M": 1}).to_numpy()
X = uci_bc_data.drop("diagnosis", axis=1).to_numpy()
# Our code
#shuffle for Q4
import sklearn as sk
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, recall_score
from texttable import Texttable
from numpy import random
import matplotlib.pyplot as plt
folds = 5
it = 100
prange = 25
# 3 measurements for 2 methods
QDAF1array = np.empty([folds,it])
DTCF1array = np.empty([folds,it])
QDAaccuracyarray = np.empty([folds,it])
DTCaccuracyarray = np.empty([folds,it])
QDAsensitivityarray = np.empty([folds,it])
DTCsensitivityarray = np.empty([folds,it])
QDAplot = np.empty([prange])
DTCplot = np.empty([prange])
for p in range(prange):
    shuffle = True
    newp = p/100
    if shuffle:
        for i in range(len(y)):
            r = random.rand()
            if r < newp:
                y[i] = 1 - y[i]
    for j in range(it):
      X_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
      #kfold = sk.model_selection.KFold(n_splits=folds, shuffle=True)
      kfold = sk.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
      QDA = QuadraticDiscriminantAnalysis()
      DTC = tree.DecisionTreeClassifier()
      for i, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)): #remove y_train when you dont use stratisfied
          newX_train = [X_train[idx] for idx in train_index]
          newy_train = [y_train[idx] for idx in train_index]
          QDA.fit(newX_train, newy_train)
          #print(QDA.predict_proba(newX_train))
          DTC.fit(newX_train, newy_train)
          newX_test = [X_train[idx] for idx in val_index]
          newy_test = [y_train[idx] for idx in val_index]
          QDAF1array[i,j] = f1_score(newy_test,QDA.predict(newX_test))
          DTCF1array[i,j] = f1_score(newy_test,DTC.predict(newX_test))

          QDAaccuracyarray[i,j] = accuracy_score(newy_test,QDA.predict(newX_test))
          DTCaccuracyarray[i,j] = accuracy_score(newy_test,DTC.predict(newX_test))

          QDAsensitivityarray[i,j] = recall_score(newy_test,QDA.predict(newX_test))
          DTCsensitivityarray[i,j] = recall_score(newy_test,DTC.predict(newX_test))
    QDAplot[p] = np.mean(QDAF1array) # change for different plots
    DTCplot[p] = np.mean(DTCF1array) # same
plt.plot(range(prange),QDAplot) #plotting
plt.plot(range(prange),DTCplot)
plt.ylabel('sensitivity')
plt.xlabel('% of mislabelled data')
plt.plot(range(prange), QDAplot, label ='QDA')
plt.plot(range(prange), DTCplot, label ='DTC/CART')
plt.legend()
plt.show()
## Results
t = Texttable() #print results for Q3
t.add_rows([['X', 'Accuracy', 'Sensitivity', 'F1-Score'],
            ['QDA', '%f with std %f' % (np.mean(QDAaccuracyarray),np.std(QDAaccuracyarray)),'%f with std %f' % (np.mean(QDAsensitivityarray),np.std(QDAsensitivityarray)), '%f with std %f' % (np.mean(QDAF1array),np.std(QDAF1array))],
            ['DTC', '%f with std %f' % (np.mean(DTCaccuracyarray), np.std(DTCaccuracyarray)), '%f with std %f' % (np.mean(DTCsensitivityarray), np.std(DTCsensitivityarray)),'%f with std %f' % (np.mean(DTCF1array), np.std(DTCF1array))]])
print(t.draw())