# -*- coding: utf-8 -*-
"""
Demonstration the importance of task division strategy in determing the Conintual Learning Accuracy
@author: Yucheng Fu
Package Version Used
python 3.9
pytorch 1.12.1
Tested in Spyder
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

RepeatTimes = 5
data= sio.loadmat('EE_Data.mat')
data = data['data'][0:]
#(0-9 Input parameters;)
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

# %% Task Dividing
set_seed(seed=SEED)
# Different Division Method
tasks_num = 5
Division_method = 'Input'
Para_Select = 0;
task_data1 = task_division(X,y, Division_method, tasks_num,Para_Select)

tasks_num = 5
Division_method = 'Input'
Para_Select = 8;
task_data2 = task_division(X,y, Division_method, tasks_num,Para_Select)

tasks_num = 5
Division_method = 'Output'
Para_Select = None;
task_data3 = task_division(X,y, Division_method, tasks_num,Para_Select)


# %% Learning without forgetting
LwF_lambda = 1;
tasks_num = 5;
accs_LwF_rep1 = LwF(InputLen, task_data1, tasks_num, LwF_lambda, RepeatTimes, DEVICE)
accs_LwF_rep2 = LwF(InputLen, task_data2, tasks_num, LwF_lambda, RepeatTimes, DEVICE)
accs_LwF_rep3 = LwF(InputLen, task_data3, tasks_num, LwF_lambda, RepeatTimes, DEVICE)

# %% EWC Method
ewc_lambda = 0.9
tasks_num = 5;
accs_ewc_rep_SH1 = EWC_SH(InputLen, task_data1, tasks_num, RepeatTimes, ewc_lambda,DEVICE)
accs_ewc_rep_SH2 = EWC_SH(InputLen, task_data2, tasks_num, RepeatTimes, ewc_lambda,DEVICE)
accs_ewc_rep_SH3 = EWC_SH(InputLen, task_data3, tasks_num, RepeatTimes, ewc_lambda,DEVICE)

# %% Results Visualizatoin
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_task_error(lists_to_plot, legend_name, save_file_name, ylim=None, figsize=(4,3)):
    plt.rcParams['font.family'] = 'Times New Roman'
    line_styles = ['-', '--', '-.', ':', '--', '-']
    marker_shapes = ['o', '^', 's', 'D', 'v', 'p']
    scatter_colors = sns.color_palette("bright", 6)

    fig, ax = plt.subplots(figsize=figsize)
    for i, lst in enumerate(lists_to_plot):
        a = np.array(lst)
        b = np.nanmean(a, axis=2)
        means = np.mean(b, axis=0)
        stds = np.std(b, axis=0)
        ax.plot(np.arange(len(means)) + 1, means, marker=marker_shapes[i], markersize=8,
                linestyle=line_styles[i], mec='black', mfc=scatter_colors[i],
                label=legend_name[i], color='black', linewidth=1)
        ax.errorbar(np.arange(len(means)) + 1, means, yerr=stds,
                    fmt='none', capsize=3, color=scatter_colors[i], zorder=1)

    ax.set_xlabel('Number of Tasks', fontsize=8)
    ax.set_ylabel('Seen Task Average Error [%]', fontsize=8)
    ax.legend(fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Apply the percentage formatter to the y-axis
    def to_percent(y, position):
        s = "{:.1f}".format(100 * y)
        return s + '%'
    formatter = plt.FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    # Define the subfolder path
    subfolder_path = 'Image and Data'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    plt.savefig(os.path.join(subfolder_path, save_file_name), dpi=300, bbox_inches='tight')
    plt.show()


legend_name = ['Divide Task by $\\alpha_n$','Divide Task by $E_n$','Dvide Task by EE'] 
SaveFigName = 'EE_EWC_Method_Compare.jpg'
lists_to_plot = [accs_ewc_rep_SH1[0], accs_ewc_rep_SH2[0], accs_ewc_rep_SH3[0]]
plot_task_error(lists_to_plot, legend_name, SaveFigName , ylim=None, figsize=(4,3))
    
    
legend_name = ['Divide Task by $\\alpha_n$','Divide Task by $E_n$','Dvide Task by EE'] 
SaveFigName = 'EE_LwF_Method_Compare.jpg'
lists_to_plot = [accs_LwF_rep1[0], accs_LwF_rep2[0], accs_LwF_rep3[0]]
plot_task_error(lists_to_plot, legend_name, SaveFigName , ylim=None, figsize=(4,3))
    