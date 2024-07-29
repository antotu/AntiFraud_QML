"""
Post processing method.
Steps:
    1. Find the threshold value for which the recall value is 1 in the training set and the precision is the highest value
    2. Save the threshold value, the recall value and the precision value in both the training and test sets.
"""
# Import section 
import pandas as pd
import torch
from VQCStronglyEntanglingModel import VQCStronglyEntanglingModel
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import os
from sklearn.metrics import recall_score, precision_score


listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])
SignificantFigures = int(listArg[3])


X_train = pd.read_csv(f"../../ReducedDataset/Train_{numElements}_{numFeatures}.csv", header=None, index_col=False)
X_test = pd.read_csv(f"../../ReducedDataset/Test_{numElements}_{numFeatures}.csv", header=None, index_col=False)
y_train = X_train.pop(2)
y_test = X_test.pop(2)
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


BATCH_SIZE = 10

train_dataloader = DataLoader(train_tensor, 
                              batch_size=BATCH_SIZE,
                              shuffle=True)


test_dataloader = DataLoader(test_tensor, 
                              batch_size=BATCH_SIZE,
                              shuffle=False)


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
        self.layer1 = VQCStronglyEntanglingModel(
            encoding=encoding,
            numLayerStrongly=numLayerStrongly,
            numClasses=2,
            numWires=numWires,
            gates=gates,
            reUploading=reuploading
        )
    # the input vector is firstly passed to the Kernel module and then to the VQC
    def forward(self,x):
        return self.layer1(x)
    

def conf_matrix(y_true, y_pred):
    res = np.zeros((2, 2))
    for pred, true in zip(y_pred, y_true):
        if true == 0 and pred == 0:
            res[0][0] += 1
        elif true == 1 and pred == 1:
            res[1][1] += 1
        elif true == 0 and pred == 1:
            res[0][1] += 1
        else:
            res[1][0] += 1
    return res

def findThreshold(y_pred, y_true):
    threshold = 0.000
    unit = 0.1

    for _ in range(SignificantFigures):
        valueOk = True
        cnt = 0
        while (valueOk):
            p = np.array([1 if (p >= (threshold + unit * (cnt + 1))) else 0 for p in y_pred])
            if recall_score(y_pred=p, y_true=y_true.numpy()) != 1:
                valueOk = False
            else:
                cnt += 1
        threshold += unit * cnt
        unit = unit / 10
    return threshold
    
        



fileRes = open("ResThresholdRecallPrecision.txt", "w")

lsFile = os.listdir()
for file in lsFile:
    if file.endswith(".pth") and file.startswith("Strongly"):
        print(file)
        MODEL_SAVE_PATH = file 
        fileName = file[:-4]
        argModel = fileName.split("_")
        # case with superposition
        if len(argModel) == 7:
            encoding = argModel[3] + "_" +argModel[4]
            numLayers = int(argModel[5])
            reup = (argModel[6] == "True")
        else:
            encoding = argModel[3]
            numLayers = int(argModel[4])
            reup = (argModel[5] == "True")



        model_0 = FraudDetection(
            numWires=numWires,
            numLayerStrongly = numLayers,
            encoding = "angle",
            gates=encoding,
            reuploading=reup
        )

        model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

        confMatrixTest = np.zeros((2,2))
        resTrain = torch.Tensor()
        with torch.inference_mode():
            for X_t, y_t in train_dataloader:
                test_logits = model_0(X_t).squeeze()
                test_pred = torch.sigmoid(test_logits)
                resTrain = torch.cat((resTrain, test_pred))

        
        resTest = torch.Tensor()
        with torch.inference_mode():
            for X_t, y_t in test_dataloader:
                test_logits = model_0(X_t).squeeze()
                test_pred = torch.sigmoid(test_logits)
                resTest = torch.cat((resTest, test_pred))
        

        
        ## individua valore soglia
        threshold = findThreshold(resTrain, y_train)
        y_pred_train = np.array([1 if p >= (threshold) else 0 for p in resTrain])
        y_pred_test = np.array([1 if p >= (threshold) else 0 for p in resTest])
        print(threshold, recall_score(y_pred=y_pred_train, y_true=y_train.numpy()),
               recall_score(y_pred=y_pred_test, y_true=y_test.numpy()),
               precision_score(y_pred=y_pred_train, y_true=y_train.numpy()),
               precision_score(y_pred=y_pred_test, y_true=y_test.numpy())
               )
        fileRes.write(fileName + "," + str(threshold) + "," + str(recall_score(y_pred=y_pred_train, y_true=y_train.numpy())) + ","
               + str(recall_score(y_pred=y_pred_test, y_true=y_test.numpy()))+ "," + 
               str(precision_score(y_pred=y_pred_train, y_true=y_train.numpy()))+ "," +
               str(precision_score(y_pred=y_pred_test, y_true=y_test.numpy())) + "\n")
        


fileRes.close()
              
