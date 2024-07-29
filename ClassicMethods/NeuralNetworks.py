
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import os
from sklearn.metrics import confusion_matrix



listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])
numHidden = int(listArg[3])

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
    
model = FraudDetection(inputs=numFeatures, hidden_units=numHidden)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))


lr = 0.01
# Set the loss function
# This is a Binary Cross Entropy with the application of the sigmoid function
loss_fn = nn.BCEWithLogitsLoss() 
# The chosen optimizer is the Adam with initial learning rate of set by variable lr
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# Set the number of epochs to 10
epochs = 10

# set the initial test loss to 0 and the accuracy to 0
test_loss = 0
test_acc = 0



def accuracy_fn(y_true, y_pred):
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correct += 1
    #correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    #print(acc)
    return acc


"Initial evaluation of the model on the test set"
with torch.inference_mode():
    for X_t, y_t in test_dataloader:
        # get the results
        test_logits = model(X_t).squeeze()
        # get the prediction value
        test_pred = torch.round(torch.sigmoid(test_logits))
        # add the loss value of the single batch to the resulting one 
        test_loss += loss_fn(test_logits, y_t)
        # sum the accuracy of the batch to the resulting one
        test_acc += accuracy_fn(y_true=y_t, y_pred=test_pred)
    # get the mean of the loss and accuracy
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f"Epoch: 0\n-----")
    print(f"Test loss: {test_loss:4f} | Test acc: {test_acc:4f}\n")



# loop for number of epochs time
for epoch in tqdm(range(epochs)):
    # set the initial loss and accuracy to 0
    train_loss = 0
    train_acc = 0
    # loop for each batch of the training dataloader
    for batch, (X_tr, y_tr) in enumerate(train_dataloader):
        # start the training procedure
        model.train()
        # get the output of the models
        y_logits = model(X_tr).squeeze()
        # get the prediction result
        y_pred = torch.round(torch.sigmoid(y_logits))
        # calculate the loss
        loss = loss_fn(y_logits, y_tr)
        # calculate the accuracy
        acc = accuracy_fn(y_true=y_tr, y_pred=y_pred)
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
    model.eval()
    with torch.inference_mode():
        # loop for each batch of the test dataloader and get 
        # the mean of test loss and test accuracy
        for X_t, y_t in test_dataloader:
            test_logits = model(X_t).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss += loss_fn(test_logits, y_t)
            test_acc += accuracy_fn(y_true=y_t, y_pred= test_pred)
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Train loss: {train_loss:4f} | Train acc: {train_acc:4f} | Test loss: {test_loss:4f} | Test acc: {test_acc:4f}")
        
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


MODEL_SAVE_PATH = f"./NeuralNetwork/NN_{numElements}_{numFeatures}_{numHidden}.pth"
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


with torch.inference_mode():
    resTrain = torch.sigmoid(model(X_train).squeeze())
    resTest = torch.sigmoid(model(X_test).squeeze())
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
        ).to_csv(f"./NeuralNetwork/ResPrecisionRecall_NeuralNetwork_{numElements}_{numFeatures}_{numHidden}.csv")


