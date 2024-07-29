import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import os
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])


X_train = pd.read_csv(f"../ReducedDataset/Train_{numElements}_{numFeatures}.csv", header=None, index_col=False)
X_test = pd.read_csv(f"../ReducedDataset/Test_{numElements}_{numFeatures}.csv", header=None, index_col=False)
y_train = X_train.pop(numFeatures)
y_test = X_test.pop(numFeatures)
y_train = torch.from_numpy(np.array(y_train)).type(torch.float)
y_test = torch.from_numpy(np.array(y_test)).type(torch.float)

minmaxScaler = MinMaxScaler(feature_range=(-1, 1))

X_train = minmaxScaler.fit_transform(X_train)
X_test = minmaxScaler.transform(X_test)


model = GradientBoostingClassifier()   
  
# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
model.fit(X_train, y_train) 
  
# performing predictions on the test dataset 
resTest = model.predict_proba(X_test) 
resTrain = model.predict_proba(X_train)

dump(model, f'./XGBoostResult/XGBoost_{numElements}_{numFeatures}.joblib')

def getRes(y_pred, y_true, threas):
    tpList = []
    fpList = []
    fnList = []
    tnList = []
    for threashold in threas:
        p = np.array([1 if p[1] >= threashold else 0 for p in y_pred])
        tn, fp, fn, tp = confusion_matrix(y_pred=p, y_true=y_true).ravel()
        tnList.append(tn)
        fpList.append(fp)
        fnList.append(fn)
        tpList.append(tp)
        
    return tpList, fpList, fnList, tnList


threas = [i/1000 for i in range(100)] + [i/10 for i in range(1, 6)]
## individua valore soglia
tpTrain, fpTrain, fnTrain, tnTrain = getRes(resTrain, y_train, threas)
tpTest, fpTest, fnTest, tnTest = getRes(resTest, y_test, threas)
pd.DataFrame({"threashold" : threas, 
                "tpTrain" : tpTrain, 
                "fpTrain" : fpTrain,
                "fnTrain" : fnTrain, 
                "tnTrain" : tnTrain,
                "tpTest" : tpTest, 
                "fpTest" : fpTest,
                "fnTest" : fnTest, 
                "tnTest" : tnTest,
            }
    ).to_csv(f"./XGBoostResult/ResPrecisionRecall_XGBoost_{numElements}_{numFeatures}.csv")



