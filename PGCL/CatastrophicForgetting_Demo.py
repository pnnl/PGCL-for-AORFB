# -*- coding: utf-8 -*-
"""
@author: Yucheng Fu
Demonstration of Catastrophic Forgetting when Using ML wihtout CL
@author: Yucheng Fu
Package Version Used
python 3.9
pytorch 1.12.1
"""
# %% Import Modules

import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import pandas as pd
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import scipy.io as sio
import seaborn as sns

'''Customize Library'''
sys.path.insert(0, './CL_PythonFunctions/')
from BaseFunctions import *
from DeviceSetting import *
from CL_Methods_Voltage import *
from Data_Division import *
import os

# %% Set random seed and device
SEED = 2023
# 2023 good so far
set_seed(seed=SEED)
DEVICE ='cpu'

save_path = 'Image and Data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% Load Training Data for Catastrophic Forgetting
RepeatTimes = 1
data= sio.loadmat('Voltage_Data.mat')
data = data['data'][0:]
data = data[np.random.choice(data.shape[0], data.shape[0], replace=False), :]
#(0-9 Input parameters, 10-14: i, Q, discharge/charge, SOC, V, CaseNum)
SelX = [0,1,2,3,4,5,6,7,8,9,12,13]
SelY = 14

InputLen = len(SelX)
X = data[:,SelX].astype(np.float32) 
y = data[:,SelY].astype(np.float32)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
# Preapre train/test data
train, test = prepare_data(X,y)

# %% Load the test data for Catastrophic Forgetting
data= sio.loadmat('CF_Test.mat')
SC = data['data'] 
InputLen = len(SelX)
X_SC = SC[:,SelX].astype(np.float32) 
X_SC = min_max_scaler.transform(X_SC)
y_SC = SC[:,SelY].astype(np.float32)
SC_test = prepare_data_test(X_SC,y_SC)


# %% Task Division for Training Dataset
set_seed(seed=SEED)
tasks_num = 5
Division_method = 'Input' # Select task division method and parameter
Para_Select = 8;
task_data = task_division(X,y, Division_method, tasks_num,Para_Select)

# %% No Regulated CNN
tasks_num = 5;
_,_, SC_pre = CNN(InputLen, task_data, tasks_num, RepeatTimes, SC_test)

# %%EWC Method
ewc_lambda = 0.001
tasks_num = 5;
_,_, accs_ewc_rep_ex_pre,_= EWC_SH(InputLen, task_data, tasks_num, RepeatTimes, ewc_lambda,DEVICE,SC_test)

# %% Plot for CNN
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties


plt.rcParams['font.family'] = 'Times New Roman'  # Font family for the plot
plt.rcParams['font.size'] = 28  # Font size for the plot

column_to_plot = X_SC[:, -1]  # Voltage

fig, ax = plt.subplots(figsize=(9, 6))  # Create a figure and axes with specified size

sorted_indices = np.argsort(column_to_plot)
sorted_column = column_to_plot[sorted_indices]
sorted_y_SC = y_SC[sorted_indices]

markers = ['^', 's', 'p', '*', 'x']
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

for i in range(5):  # Looping through 5 curves, assuming 0 to 4 inclusive
    ax.scatter(column_to_plot, SC_pre[0][i], label=f'$DNN_{i+1}$', 
                marker=markers[i], color=colors[i], s=64)  # Different marker and color for each curve
threshold = 1.62
above_threshold = sorted_y_SC > threshold
below_threshold = sorted_y_SC <= threshold
ax.scatter(sorted_column[above_threshold], sorted_y_SC[above_threshold], facecolors='none', edgecolors='black', marker='o', s=100,linewidths=2, label='Ground Truth')
ax.plot(sorted_column[above_threshold], sorted_y_SC[above_threshold], color='black', linestyle='dashed')
ax.scatter(sorted_column[below_threshold], sorted_y_SC[below_threshold], facecolors='none', edgecolors='black', marker='o', s=100, linewidths=2,label='')
ax.plot(sorted_column[below_threshold], sorted_y_SC[below_threshold], color='black', linestyle='dashed')

font_prop = FontProperties(family='Times New Roman', size=16)
ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1), prop=font_prop)

plt.tight_layout()
ax.set_xlabel('SOC')
ax.set_ylabel('Voltage [V]')
ax.minorticks_on()
ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
ax.tick_params(direction='in', length=8, width=2, which='major')
ax.tick_params(direction='in', length=8, width=1, which='minor')
for spine in ax.spines.values():
    spine.set_linewidth(2)
# Set major and minor tick intervals
ax.xaxis.set_major_locator(MultipleLocator(0.3))  # Adjust major x ticks interval
ax.yaxis.set_major_locator(MultipleLocator(0.3))  # Adjust major y ticks interval
ax.xaxis.set_minor_locator(MultipleLocator(0.1))  # Adjust minor x ticks interval (smaller value for finer control)
ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Adjust minor y ticks interval
# Draw grid lines
ax.grid(True, which='both', linestyle='-', linewidth=0.7)

ax.set_xlim([0, 1.1])
ax.set_ylim([1, 2])
plt.show()


# %% Plot for EWC

plt.rcParams['font.family'] = 'Times New Roman'  # Font family for the plot
plt.rcParams['font.size'] = 28  # Font size for the plot

column_to_plot = X_SC[:, -1]  # Voltage

fig, ax = plt.subplots(figsize=(9, 6))  # Create a figure and axes with specified size

sorted_indices = np.argsort(column_to_plot)
sorted_column = column_to_plot[sorted_indices]
sorted_y_SC = y_SC[sorted_indices]

markers = ['^', 's', 'p', '*', 'x']
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

for i in range(5):  # Looping through 5 curves, assuming 0 to 4 inclusive
    ax.scatter(column_to_plot, accs_ewc_rep_ex_pre[0][i], label=f'$EWC_{i+1}$', 
                marker=markers[i], color=colors[i], s=64,edgecolors='black')  # Different marker and color for each curve
threshold = 1.62
above_threshold = sorted_y_SC > threshold
below_threshold = sorted_y_SC <= threshold
ax.scatter(sorted_column[above_threshold], sorted_y_SC[above_threshold], facecolors='none', edgecolors='black', marker='o', s=100,linewidths=2, label='Ground Truth')
ax.plot(sorted_column[above_threshold], sorted_y_SC[above_threshold], color='black', linestyle='dashed')
ax.scatter(sorted_column[below_threshold], sorted_y_SC[below_threshold], facecolors='none', edgecolors='black', marker='o', s=100, linewidths=2,label='')
ax.plot(sorted_column[below_threshold], sorted_y_SC[below_threshold], color='black', linestyle='dashed')


font_prop = FontProperties(family='Times New Roman', size=16)
ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1), prop=font_prop)

plt.tight_layout()
ax.set_xlabel('SOC')
ax.set_ylabel('Voltage [V]')
ax.minorticks_on()
ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
ax.tick_params(direction='in', length=8, width=2, which='major')
ax.tick_params(direction='in', length=8, width=1, which='minor')
for spine in ax.spines.values():
    spine.set_linewidth(2)
# Set major and minor tick intervals
ax.xaxis.set_major_locator(MultipleLocator(0.3))  # Adjust major x ticks interval
ax.yaxis.set_major_locator(MultipleLocator(0.3))  # Adjust major y ticks interval
ax.xaxis.set_minor_locator(MultipleLocator(0.1))  # Adjust minor x ticks interval (smaller value for finer control)
ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Adjust minor y ticks interval
# Draw grid lines
ax.grid(True, which='both', linestyle='-', linewidth=0.7)

ax.set_xlim([0, 1.1])
ax.set_ylim([1, 2])
plt.show()


