import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn import tree
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
QDAscore=0
DTCscore=0
for j in range(10):
  X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
  kfold = sk.model_selection.KFold(n_splits=folds)
  #kfold = sk.model_selection.StratifiedKFold(n_splits=folds)
  QDAresult_array = np.empty([folds,4])
  DTCresult_array = np.empty([folds,4])
  QDA = QuadraticDiscriminantAnalysis()
  DTC = tree.DecisionTreeClassifier()
  for i, (train_index, val_index) in enumerate(kfold.split(X_train)): #remove y_rain when you dont use stratisfied
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
  QDAscore = QDAscore+QDAmean[3]
  DTCscore = DTCscore+DTCmean[3]
print(QDAscore/len(range(10)))
print(DTCscore/len(range(10)))