import numpy as np
import random


n = 4


#X=np.random.random((n,2))
X = np.zeros([n,2])
y = np.zeros(n)

##
for i in range(n):
        X[i,0] = random.uniform(0,10)
        X[i,1] = random.uniform(0,10)
        if X[i,1] <=4 and X[i,0] >=5:
            y[i] = 1
      #  else:  #Not needed
       #     y[i] = 0

