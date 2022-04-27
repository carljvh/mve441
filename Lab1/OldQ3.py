# Code from question 3
# gammal kod; ignorara for now
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
gammal kod
QDAscore=np.empty([1,4])
DTCscore=np.empty([1,4])
QDAarray=np.empty([it,4])
DTCarray=np.empty([it,4])
  QDAresult_array = np.empty([folds,4])
  DTCresult_array = np.empty([folds,4])
      QDAresult_array[i,:] = testc(newy_test,QDA.predict(newX_test))
      DTCresult_array[i,:] = testc(newy_test,DTC.predict(newX_test))
  QDAmean = np.mean(QDAresult_array.astype(int), axis=0)
  DTCmean = np.mean(DTCresult_array.astype(int), axis=0)
  QDAscore = QDAscore+QDAmean
  DTCscore = DTCscore+DTCmean
QDAscore=QDAscore/it
DTCscore=DTCscore/it