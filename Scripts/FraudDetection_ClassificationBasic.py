# Breast classification 
import pandas as pd
import torch
from VQCBasicEntanglingModel import VQCBasicEntanglingLayerModel

# Load data and preprocessing

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
import requests
import telegram_send
import sys


listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])
encoding = listArg[3]
numLayers = int(listArg[4])
reup = bool(int(listArg[5]))


#telegram settings: Add your info
TOKEN = ""
chat_id = ""

X_train = pd.read_csv(f"../ReducedDataset/Train_{numElements}_{numFeatures}.csv", header=None, index_col=False)
X_test = pd.read_csv(f"../ReducedDataset/Test_{numElements}_{numFeatures}.csv", header=None, index_col=False)
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

message = f"Basic,{numElements},{numFeatures},{encoding},{numLayers},{reup} starts"    
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
requests.get(url).json()
# define the model class
class FraudDetection(nn.Module):
    # parameters which can be set:
    # numLayer (int) : set the number of layer for Basic Entangling ansatz for the VQC model
    # encoding (str): set the type of encoding to use (accepted "angle" or "amplitude")
    # gates (str): set the type of encoding gates used for the VQC  
    def __init__(self,
                numWires : int,
                numLayer: int,
                encoding: str,
                gates: str,
                reuploading: bool):
        super().__init__()
        # for the VQC, the number of classes is fixed to 2 considering the binary classification
        # the number of wires is equal to 5 resulting input vector is composed of 5 elements
        # reuploading in this case is not applied
        self.layer1 = VQCBasicEntanglingLayerModel(
            encoding=encoding,
            numLayer=numLayer,
            numClasses=2,
            numWires=numWires,
            gates=gates,
            reUploading=reuploading
        )
    # the input vector is firstly passed to the Kernel module and then to the VQC
    def forward(self,x):
        return self.layer1(x)
# set the model
model_0 = FraudDetection(
    numWires=numWires,
    numLayer = numLayers,
    encoding = "angle",
    gates=encoding,
    reuploading=reup
)
print(list(model_0.parameters()))
def accuracy_fn(y_true, y_pred):
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correct += 1
    #correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    #print(acc)
    return acc

lr = 0.01
# Set the loss function
# This is a Binary Cross Entropy with the application of the sigmoid function
loss_fn = nn.BCEWithLogitsLoss() 
# The chosen optimizer is the Adam with initial learning rate of set by variable lr
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=lr)

# Set the number of epochs to 10
epochs = 10

# set the initial test loss to 0 and the accuracy to 0
test_loss = 0
test_acc = 0

"Initial evaluation of the model on the test set"
with torch.inference_mode():
    for X_test, y_test in test_dataloader:
        # get the results
        test_logits = model_0(X_test).squeeze()
        # get the prediction value
        test_pred = torch.round(torch.sigmoid(test_logits))
        # add the loss value of the single batch to the resulting one 
        test_loss += loss_fn(test_logits, y_test)
        # sum the accuracy of the batch to the resulting one
        test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred)
    # get the mean of the loss and accuracy
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f"Epoch: 0\n-----")
    print(f"Test loss: {test_loss:4f} | Test acc: {test_acc:4f}")


    #message = f"Basic,{numElements},{numFeatures},{encoding},{numLayers},{reup} | Test loss: {test_loss:4f} | Test acc: {test_acc:4f}"    
    #url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    #requests.get(url).json()

# loop for number of epochs time
for epoch in tqdm(range(epochs)):
    # set the initial loss and accuracy to 0
    train_loss = 0
    train_acc = 0
    # loop for each batch of the training dataloader
    for batch, (X, y) in enumerate(train_dataloader):
        # start the training procedure
        model_0.train()
        # get the output of the models
        y_logits = model_0(X).squeeze()
        # get the prediction result
        y_pred = torch.round(torch.sigmoid(y_logits))
        # calculate the loss
        loss = loss_fn(y_logits, y)
        # calculate the accuracy
        acc = accuracy_fn(y_true=y, y_pred= y_pred)
        # sum the values
        train_loss += loss
        train_acc += acc
        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f"{batch * len(X)}/ {len(train_dataloader.dataset)} samples")
        
    # calculate the mean of the training loss anche training accuracy 
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    # set the initial test loss and test accuracy to 0
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        # loop for each batch of the test dataloader and get 
        # the mean of test loss and test accuracy
        for X_test, y_test in test_dataloader:
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss += loss_fn(test_logits, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred= test_pred)
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        #print(f"Epoch: {epoch + 1}\n-----")
        
        print(f"Train loss: {train_loss:4f} | Train acc: {train_acc:4f} | Test loss: {test_loss:4f} | Test acc: {test_acc:4f}")
        #message = f"Epoch {epoch} | Basic,{numElements},{numFeatures},{encoding},{numLayers},{reup} | Train loss: {train_loss:4f} | Train acc: {train_acc:4f} | Test loss: {test_loss:4f} | Test acc: {test_acc:4f}"    
        #url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        #requests.get(url).json()
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

message = f"Epoch {epoch} | Basic,{numElements},{numFeatures},{encoding},{numLayers},{reup} | Train loss: {train_loss:4f} | Train acc: {train_acc:4f} | Test loss: {test_loss:4f} | Test acc: {test_acc:4f}"    
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
requests.get(url).json()



MODEL_SAVE_PATH = f"Basic_{numElements}_{numFeatures}_{encoding}_{numLayers}_{reup}.pth"
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

"""
confMatrixTest = np.zeros((2,2))
with torch.inference_mode():
    for X_test, y_test in test_dataloader:
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        confMatrixTest += conf_matrix(y_pred=test_pred.numpy(), y_true=y_test.numpy())
        
        
print(confMatrixTest)

fileTest = open(f"confTestBasic_{numElements}_{numFeatures}_{encoding}_{numLayers}_{reup}.csv", "w")
#fileTest.write(name_conf + ",")
np.savetxt(fileTest, confMatrixTest, fmt="%d", delimiter=",", newline=",")
fileTest.write("\n")
"""


message = f"Basic,{numElements},{numFeatures},{encoding},{numLayers},{reup} ends"    
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
requests.get(url).json()