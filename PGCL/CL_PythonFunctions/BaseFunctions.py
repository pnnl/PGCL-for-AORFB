# -*- coding: utf-8 -*-
"""
Base data prepare and ML modules
@author:Yucheng Fu
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from numpy import vstack
from sklearn.metrics import mean_squared_error
# %% Dataset Processing
# 
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self,X,y):
        # store the inputs and outputs
        self.X = X
        self.y = y
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
# prepare the dataset
def prepare_data(X,y):
    dataset = CSVDataset(X,y)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

class CSVDataset_test(Dataset):
    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = y
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get all data for testing (no shuffle)
    def get_test_data(self):
        return Subset(self, range(len(self.X)))
    
# Modify prepare_data to create a test data loader without shuffling
def prepare_data_test(X, y, batch_size=32):
    # Create a TensorDataset from X and y
    test = CSVDataset_test(X, y)
    # Create the test data loader without shuffling
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return test_dl

#%% Base Function Definition

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 20)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(20, 20)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(20, 5)
        self.act3 = ReLU()
        self.output_layer = Linear(5,1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.output_layer(X)
        return X
    
    
class MLP_MH(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP_MH, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 20)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(20, 20)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(20, 5)
        self.act3 = ReLU()
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X


## Output layer, which will be separate for each task
class FHeadNet(nn.Module):
  def __init__(self, base_net, input_size=5):
    super(FHeadNet, self).__init__()
    self.base_net = base_net
    self.output_layer = nn.Linear(input_size, 1)
    xavier_uniform_(self.output_layer.weight)
    
  def forward(self, x):
    x = self.base_net.forward(x)
    x = self.output_layer(x)
    return x


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
    # print(f"Train Epoch: {epoch} \tLoss: {loss.item():.2e}")

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse, predictions, actuals

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
