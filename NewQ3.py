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
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import tree
from texttable import Texttable
def testc(y_actual, y): #returns all true-positives, false positives etc for our classifications
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if y_actual[i]==y[i]==1:
           TP += 1
        if y[i]==1 and y_actual[i]!=y[i]:
           FP += 1
        if y_actual[i]==y[i]==0:
           TN += 1
        if y[i]==0 and y_actual[i]!=y[i]:
           FN += 1
    return(TP, FP, TN, FN)
folds=10
QDAscore=np.empty([1,4])
DTCscore=np.empty([1,4])
it=10
for j in range(it):
  X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
  #kfold = sk.model_selection.KFold(n_splits=folds, shuffle=True)
  kfold = sk.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
  QDAresult_array = np.empty([folds,4])
  DTCresult_array = np.empty([folds,4])
  QDA = QuadraticDiscriminantAnalysis()
  DTC = tree.DecisionTreeClassifier()
  for i, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)): #remove y_rain when you dont use stratisfied
      newX_train = [X_train[idx] for idx in train_index]
      newy_train = [y_train[idx] for idx in train_index]
      QDA.fit(newX_train, newy_train)
      DTC.fit(newX_train, newy_train)
      newX_test = [X_train[idx] for idx in val_index]
      newy_test = [y_train[idx] for idx in val_index]

      QDAresult_array[i,:] = testc(newy_test,QDA.predict(newX_test))
      DTCresult_array[i,:] = testc(newy_test,DTC.predict(newX_test))

  QDAmean = np.mean(QDAresult_array.astype(int), axis=0)
  DTCmean = np.mean(DTCresult_array.astype(int), axis=0)
  QDAscore = QDAscore+QDAmean
  DTCscore = DTCscore+DTCmean

QDAscore=QDAscore/it
DTCscore=DTCscore/it

## Results
t = Texttable()
t.add_rows([['X', 'Accuracy', 'Sensitivity', 'F1-Score'],
            ['QDA', (QDAscore[0,2]+QDAscore[0,0])/(QDAscore[0,2]+QDAscore[0,0]+QDAscore[0,1]+QDAscore[0,3]),QDAscore[0,0]/(QDAscore[0,0]+QDAscore[0,3]), 2*QDAscore[0,0]/(2*QDAscore[0,0]+QDAscore[0,3]+QDAscore[0,1])],
            ['DTC', (DTCscore[0,2]+DTCscore[0,0])/(DTCscore[0,2]+DTCscore[0,0]+DTCscore[0,1]+DTCscore[0,3]),DTCscore[0,0]/(DTCscore[0,0]+DTCscore[0,3]),2*DTCscore[0,0]/(2*DTCscore[0,0]+DTCscore[0,3]+DTCscore[0,1])]])
print(t.draw())