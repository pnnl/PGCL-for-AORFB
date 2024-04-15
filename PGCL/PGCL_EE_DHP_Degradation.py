# -*- coding: utf-8 -*-
"""
Testing the PGCL capability for predicting unseen ASO materials during hypothetical degradation. 
Using the DHP isomer as an example
@author: Yucheng Fu
Package Version Used
python 3.9
pytorch 1.12.1
IDE Spyder
"""
# %% Import Modules
import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import numpy as np
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


import pandas as pd
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
from CL_Methods import *
from Data_Division import *
import os

# %% Set random seed and device
SEED = 2023
set_seed(seed=SEED)
DEVICE ='cpu'

save_path = 'Image and Data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% Prepare Training Data

RepeatTimes = 1
data= sio.loadmat('EE_Data.mat')
data = data['data'][0:]
#(0-9 Input parameters;)
Param_Names = [r'$\alpha_n$', 'a', r'$\sigma_m$', r'$k_n$', r'$km_n$',
                r'$C_n$', r'$D_n$', r'$\mu_n$', r'$E_n$', r'$k_{\mathrm{eff},n}$']

SelX = [0,1,2,3,4,5,6,7,8,9]
Sel_Log = []
SelY = 10
InputLen = len(SelX)
X = data[:,SelX].astype(np.float32) 
X[:,Sel_Log] = np.log10(X[:,Sel_Log]).astype(np.float32) 
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

min_max_scaler2 = preprocessing.MinMaxScaler()
y = data[:,SelY].astype(np.float32)
train, test = prepare_data(X,y)

# %% Load Hypothetical DHP18 Degradation Data for Testing
# Parameters: alpha_n,	a,	sigma_m	,k_n,	km_n,	C_n,	D_n,	mu_n,	E_n	,k_eff,n
data_DHP= sio.loadmat('Degradation_Test.mat')
Phenazine_data = data_DHP['DHP18'];
Cycles = data_DHP['Cycles'].squeeze();

Phenazine_test_X = np.array(Phenazine_data)[:, :-1].astype(np.float32)
Phenazine_test_y = np.array(Phenazine_data)[:, -1].astype(np.float32)
Phenazine_test_X = min_max_scaler.transform(Phenazine_test_X)

Phenazine_test = prepare_data_test(Phenazine_test_X,Phenazine_test_y)


# %% Task Division
set_seed(seed=SEED)

tasks_num = 9
Division_method = 'TwoInputs'
Para_Select = [8,5]; #  E_n  and c_n
task_data = task_division(X,y, Division_method, tasks_num,Para_Select)

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Extract input and target data from DataLoader objects
task_inputs = []
task_targets = []
for train, _ in task_data:
    X_list = []
    y_list = []
    for inputs, targets in train:
        X_list.append(min_max_scaler.inverse_transform(inputs.numpy()))
        y_list.append(targets.numpy())
    task_inputs.append(np.vstack(X_list))
    task_targets.append(np.vstack(y_list))

task_inputs_ex = []
task_targets_ex = []
X_list = []
y_list = []
for inputs, targets in Phenazine_test:
    X_list.append(min_max_scaler.inverse_transform(inputs.numpy()))
    y_list.append(targets.numpy())
task_inputs_ex.append(np.vstack(X_list))
task_targets_ex.append(np.vstack(y_list))

# %% Degradation Path Visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Assumed Degradation Path
sigma_m = Phenazine_data[:, 8]  
C_n = Phenazine_data[:, 5]       

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
colors = ['blue', 'red', 'green']
markers = ['o', '^', 's']

# Define the ranges for the three sets
sets = {
    'Test case 1': range(0, 20),
    'Test case 2': range(20, 40),
    'Test case 3': range(40, 60)
}


fig, ax = plt.subplots(figsize=(8, 6))

for i, (label, indices) in enumerate(sets.items()):
    ax.scatter(sigma_m[indices], C_n[indices], label=label, color=colors[i], marker=markers[i], edgecolor='black', s=50) 
    ax.plot(sigma_m[indices], C_n[indices], linestyle='-', color=colors[i], alpha=0.5)

# Set labels and title using the specified font
ax.set_xlabel('$E_n$', fontsize=20)
ax.set_ylabel('$C_n$', fontsize=20)
ax.minorticks_on()
ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
ax.tick_params(direction='in', length=8, width=2, which='major')
ax.tick_params(direction='in', length=8, width=1, which='minor')
for spine in ax.spines.values():
    spine.set_linewidth(2)
    
# Set major and minor tick intervals
ax.xaxis.set_major_locator(MultipleLocator(0.1))  
ax.yaxis.set_major_locator(MultipleLocator(50))  
ax.xaxis.set_minor_locator(MultipleLocator(0.025))  
ax.yaxis.set_minor_locator(MultipleLocator(25))  
ax.grid(True, which='both', linestyle='-', linewidth=0.7)
ax.legend()
ax.set_ylim([380, 600])
ax.set_xlim([-1.15, -0.85])
plt.show()



# %% Train the EWC Model
RepeatTimes = 10
ewc_lambda = 0.9
accs_ewc_rep_int, accs_ewc_rep_SH_ex, accs_ewc_rep_SH_ex_pre, times_ewc_rep_int = EWC_SH_Interval(InputLen, task_data, len(task_data), RepeatTimes, ewc_lambda,DEVICE,Phenazine_test)

# %% Plot the PGCL EWC Predicition Results
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

b = np.nanmean(accs_ewc_rep_SH_ex_pre, axis=0).squeeze()
std_dev = np.nanstd(accs_ewc_rep_SH_ex_pre, axis=0).squeeze()

colors = ['blue', 'red', 'green']
markers = ['o', '^', 's']

sets = {
    'Test case 1': range(0, 20),
    'Test case 2': range(20, 40),
    'Test case 3': range(40, 60)
}

plt.figure(figsize=(15, 5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

for i, (label, indices) in enumerate(sets.items(), start=1):
    ax = plt.subplot(1, 3, i)  
    # Slice for the current set
    current_b_values = b[8, indices]
    current_std_dev = std_dev[8, indices]
    current_cycles = Cycles
    current_phenazine = Phenazine_test_y[indices]
    ax.plot(current_cycles, current_phenazine, 'ko', label='Ground Truth')
    ax.errorbar(current_cycles, current_b_values, yerr=current_std_dev, fmt=markers[i-1], color=colors[i-1], label='Predictions', capsize=5)
    ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
    ax.tick_params(direction='in', length=8, width=2, which='major')
    ax.tick_params(direction='in', length=8, width=1, which='minor')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, which='both', linestyle='-', linewidth=0.7)
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('EE')
    ax.set_title(label)
    ax.set_ylim(0.5, 1)  
    ax.legend()

plt.tight_layout()  
plt.show()





