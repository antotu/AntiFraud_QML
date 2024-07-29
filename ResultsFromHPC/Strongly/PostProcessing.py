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
from sklearn.metrics import confusion_matrix



listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])
typeAnsatz = listArg[3]

if ((typeAnsatz != "Strongly") and (typeAnsatz != "Basic")):
    print("Error in the type of ansatz")
    exit(1)

print(numElements, numFeatures, typeAnsatz)

X_train = pd.read_csv(f"../../ReducedDataset/Train_{numElements}_{numFeatures}.csv", header=None, index_col=False)
X_test = pd.read_csv(f"../../ReducedDataset/Test_{numElements}_{numFeatures}.csv", header=None, index_col=False)
y_train = X_train.pop(numFeatures)
y_test = X_test.pop(numFeatures)
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



def getRes(y_pred, y_true, threas):
    tpList = []
    fpList = []
    fnList = []
    tnList = []
    for threashold in threas:
        p = np.array([1 if p >= threashold else 0 for p in y_pred])
        tn, fp, fn, tp = confusion_matrix(y_pred=p, y_true=y_true).ravel()
        tnList.append(tn)
        fpList.append(fp)
        fnList.append(fn)
        tpList.append(tp)
        
    return tpList, fpList, fnList, tnList
    

os.chdir(f"./Features{numFeatures}")            

os.system("pwd")
lsFile = os.listdir()
for file in lsFile:
    if file.endswith(".pth") and file.startswith(typeAnsatz + "_" + f"{numElements}_{numFeatures}_"):
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

        #fileRes = open("Res_Precision_Recall.csv")
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
        ).to_csv(f"../PostProcessing/ResPrecisionRecall_{typeAnsatz}_{numElements}_{numFeatures}_{encoding}_{numLayers}_{reup}.csv")



