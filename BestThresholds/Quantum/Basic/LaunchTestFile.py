# open the json file Elements.json and read the field Encoding, Ansatz, Reuploading, NumLayers, NumElements, NumFeatures, Threshold

import json
from VQCBasicEntanglingModel import VQCBasicEntanglingLayerModel
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from sklearn.metrics import confusion_matrix
def getRes(y_pred, y_true):
    
    tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y_true).ravel()
    return tn, fp, fn, tp

# define the model class
class FraudDetection(nn.Module):
    # parameters which can be set:
    # numLayerStrongly (int) : set the number of layer for stronglyEntangling ansatz for the VQC model
    # encoding (str): set the type of encoding to use (accepted "angle" or "amplitude")
    # gates (str): set the type of encoding gates used for the VQC  
    def __init__(self,
                numWires : int,
                numLayerStrongly: int,
                encoding: str,
                gates: str,
                reuploading: bool):
        super().__init__()
        # for the VQC, the number of classes is fixed to 2 considering the binary classification
        # the number of wires is equal to 5 resulting input vector is composed of 5 elements
        # reuploading in this case is not applied
        self.layer1 = VQCBasicEntanglingLayerModel(
            encoding=encoding,
            numLayer=numLayerStrongly,
            numClasses=2,
            numWires=numWires,
            gates=gates,
            reUploading=reuploading
        )
    # the input vector is firstly passed to the Kernel module and then to the VQC
    def forward(self,x):
        return self.layer1(x)
allRes = []
with open("Elements_Train.json", "r") as file:
    data = json.load(file)
    for el in data:
        Encoding = el["Encoding"]
        Ansatz = el["Ansatz"]
        NumLayers = el["NumLayers"]
        Reuploading = el["Reuploading"]
        NumElements = el["NumElements"]
        NumFeatures = el["NumFeatures"]
        Threshold = el["Threshold"]

        MODEL_SAVE_PATH = f"../../../ResultsFromHPC/{Ansatz}/Features{NumFeatures}/"  + Ansatz + "_" + str(NumElements) + "_" + str(NumFeatures) + "_" + Encoding + "_"  + str(NumLayers) + "_" + str(Reuploading) + ".pth"
        model = FraudDetection(encoding="angle", numLayerStrongly=NumLayers, numWires=NumFeatures, reuploading=Reuploading, gates=Encoding)
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        FOLDER_DATASET = "../../../ReducedDataset/"
        FILE_TEST = f"Verification_{NumElements}_{NumFeatures}.csv"
        FILE_TRAIN = f"Train_{NumElements}_{NumFeatures}.csv"
        X_train = pd.read_csv(FOLDER_DATASET + FILE_TEST, header=None, index_col=False)
        X_test = pd.read_csv(FOLDER_DATASET + FILE_TEST, header=None, index_col=False)
        y_train = X_train.pop(NumFeatures)
        y_test = X_test.pop(NumFeatures)
        y_train = torch.from_numpy(np.array(y_train)).type(torch.float)
        y_test = torch.from_numpy(np.array(y_test)).type(torch.float)

        minmaxScaler = MinMaxScaler(feature_range=(-1, 1))

        X_train = minmaxScaler.fit_transform(X_train)
        X_test = minmaxScaler.transform(X_test)


        numWires = X_train.shape[1]

        X_train = torch.from_numpy(X_train).type(torch.float)
        X_test = torch.from_numpy(X_test).type(torch.float)
        train_tensor = torch.utils.data.TensorDataset(X_train, y_train)
        test_tensor = torch.utils.data.TensorDataset(X_test, y_test)

        model.eval()
        res = torch.sigmoid(model(X_test))
        print(res)
        prediction = [1 if p > Threshold else 0 for p in res]
        tp, fp, fn, tn = confusion_matrix(y_pred=prediction, y_true=y_test).ravel()
        print(tp, fp, fn, tn)
        dictRes = el.copy()
        dictRes["tp"] = tp
        dictRes["fp"] = fp
        dictRes["fn"] = fn
        dictRes["tn"] = tn
        allRes.append(dictRes)

# Convert int64 objects to regular Python integers
for res in allRes:
    for key, value in res.items():
        if isinstance(value, np.int64):
            res[key] = int(value)

# Serialize the data to JSON
with open(f"Results_Train.json", "w") as file:
    json.dump(allRes, file)
        


