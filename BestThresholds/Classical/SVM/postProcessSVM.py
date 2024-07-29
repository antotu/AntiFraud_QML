from joblib import load
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

hyperparam = ["linear", "linearTrain", "rbf", "rbfTrain"]
# Open and read the JSON file

def getRes(y_pred, y_true, threas):
    
    p = np.array([1 if p[1] >= threas else 0 for p in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_pred=p, y_true=y_true).ravel()
    
    return tp, fp, fn, tn

ResVal = []
for h in hyperparam:
    with open('Elements_SVM_' + h + '.json', 'r') as file:
        dataV = json.load(file)
        for data in dataV:
            numEl = data["NumElements"]
            numF = data["NumFeatures"]
            th = float(data["Threshold"])
            if h.endswith("Train"):
                hKernel = h[:-5]
            else:
                hKernel = h
            loaded_model = load(f'../../../ClassicMethods/SVMResult/SVM_{numEl}_{numF}_{hKernel}.joblib')
            DATASET_PATH = f"../../../ReducedDataset/"
            VAL_FILE = f"Verification_{numEl}_{numF}.csv"
            TRAIN_FILE = f"Train_{numEl}_{numF}.csv"
            X_train = pd.read_csv(DATASET_PATH + TRAIN_FILE, header=None, index_col=False)
            X_test = pd.read_csv(DATASET_PATH + VAL_FILE, header=None, index_col=False)
            y_train = X_train.pop(numF)
            y_test = X_test.pop(numF)
            y_train = torch.from_numpy(np.array(y_train)).type(torch.float)
            y_test = torch.from_numpy(np.array(y_test)).type(torch.float)

            minmaxScaler = MinMaxScaler(feature_range=(-1, 1))
            X_train = minmaxScaler.fit_transform(X_train)
            X_test = minmaxScaler.transform(X_test)
            y_pred = loaded_model.predict_proba(X_test)
            tpTest, fpTest, fnTest, tnTest = getRes(y_pred, y_test, th)
            ResVal.append({
                "NumElements": numEl,
                "NumFeatures" : numF,
                "kernel" : h,
                "threashold" : th, 
                "tpTest" : tpTest, 
                "fpTest" : fpTest,
                "fnTest" : fnTest, 
                "tnTest" : tnTest,
                })
# Convert int64 objects to regular Python integers
for res in ResVal:
    for key, value in res.items():
        if isinstance(value, np.int64):
            res[key] = int(value)        
# Serialize the data to JSON
with open(f"ResultsSVM.json", "w") as file:
    json.dump(ResVal, file)

