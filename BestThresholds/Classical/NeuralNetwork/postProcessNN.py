
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from torch import nn
hyperparam = ["NN", "NNTrain"]
# Open and read the JSON file

def getRes(y_pred, y_true, threas):
    
    p = np.array([1 if p >= threas else 0 for p in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_pred=p, y_true=y_true).ravel()
    
    return tp, fp, fn, tn

ResVal = []



# define the model class
class FraudDetection(nn.Module):
    # parameters which can be set:
    # numLayerStrongly (int) : set the number of layer for stronglyEntangling ansatz for the VQC model
    # encoding (str): set the type of encoding to use (accepted "angle" or "amplitude")
    # gates (str): set the type of encoding gates used for the VQC  
    def __init__(self,
                inputs : int,
                hidden_units : int):
        super().__init__()
        #define a neural network with 2 hidden layers
        self.layer1 = nn.Sequential(
            nn.Linear(inputs, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
    def forward(self,x):
        print(x.shape)
        return self.layer1(x)



for h in hyperparam:
    with open('Elements_' + h + '.json', 'r') as file:
        dataV = json.load(file)
        for data in dataV:
            numEl = data["NumElements"]
            numF = data["NumFeatures"]
            NumLayers = int(data["NumLayers"])
            th = float(data["Threshold"])
            MODEL_PATH = f"../../../ClassicMethods/NeuralNetwork/NN_{numEl}_{numF}_{NumLayers}.pth"
            model = FraudDetection(inputs=numF, hidden_units=NumLayers)
            model.load_state_dict(torch.load(f=MODEL_PATH))
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
            X_test = torch.from_numpy(X_test).type(torch.float)
            model.eval()
            #print(X_test.shape)
            print(NumLayers)
            y_pred = torch.sigmoid(model(X_test))
            print(y_pred)
            
            tpTest, fpTest, fnTest, tnTest = getRes(y_pred, y_test, th)
            ResVal.append({
                "NumElements": numEl,
                "NumFeatures" : numF,
                "NumUnits" : NumLayers,
                "threashold" : th,
                "Train" : h, 
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
with open(f"Results_NeuralNetwork.json", "w") as file:
    json.dump(ResVal, file)

