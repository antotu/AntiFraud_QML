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
from sklearn.linear_model import LogisticRegression

listArg = sys.argv
numElements = int(listArg[1])
numFeatures = int(listArg[2])


X_train = pd.read_csv(f"../../ReducedDataset/Train_{numElements}_{numFeatures}.csv", header=None, index_col=False)
X_test = pd.read_csv(f"../../ReducedDataset/Test_{numElements}_{numFeatures}.csv", header=None, index_col=False)
y_train = X_train.pop(2)
y_test = X_test.pop(2)
y_train = torch.from_numpy(np.array(y_train)).type(torch.float)
y_test = torch.from_numpy(np.array(y_test)).type(torch.float)

minmaxScaler = MinMaxScaler(feature_range=(-1, 1))

X_train = minmaxScaler.fit_transform(X_train)
X_test = minmaxScaler.transform(X_test)

model = LogisticRegression()   
  
# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
model.fit(X_train, y_train) 
  
# performing predictions on the test dataset 
y_pred = model.predict_proba(X_test) 

