# -*- coding: utf-8 -*-
"""
Testing PGCL by Task Division of 16, two input parameters
@author: Yucheng Fu
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
from matplotlib.patches import Ellipse
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
data= sio.loadmat('EE_Data.mat')
data = data['data'][0:]
#(0-9 Input parameters;)
# Enable the use of LaTeX for text formatting
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


# %% Task Dividing and Visualization
set_seed(seed=SEED)
tasks_num = 16
Division_method = 'TwoInputs'
Para_Select = [0,2]; 
task_data = task_division(X,y, Division_method, tasks_num,Para_Select)

# Specify the PGCL Task Head.
PGCL_taskcreation = [0,3,12]

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

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

# Create a colormap function using the viridis colormap
colormap = plt.cm.autumn

# Visualize the input distribution for the selected parameters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.1, 0.1, 0.6, 0.6])  # [left, bottom, width, height]

global_x_min = min([task_X[:, Para_Select[0]].min() for task_X in task_inputs])
global_x_max = max([task_X[:, Para_Select[0]].max() for task_X in task_inputs])
global_y_min = min([task_X[:, Para_Select[1]].min() for task_X in task_inputs])
global_y_max = max([task_X[:, Para_Select[1]].max() for task_X in task_inputs])
num_tasks_per_row = int(np.sqrt(tasks_num))
num_tasks_per_col = int(np.ceil(tasks_num / num_tasks_per_row))
box_width = (global_x_max - global_x_min) / num_tasks_per_row
box_height = (global_y_max - global_y_min) / num_tasks_per_col
pcolor1 = np.array([255, 202, 137]) / 255
pcolor2 = np.array([181,208,255]) / 255
for i, (task_X, task_y) in enumerate(zip(task_inputs, task_targets)):
    colors = colormap(task_y / 1)
    sc = ax.scatter(task_X[:, Para_Select[0]], task_X[:, Para_Select[1]], alpha=1, c=colors, edgecolors='black', s=80)
    box_x_min = global_x_min + (i // num_tasks_per_row) * box_width
    box_y_min = global_y_min + (i % num_tasks_per_row) * box_height
    
    # Draw a dashed box for each task
    ax.plot([box_x_min, box_x_min + box_width, box_x_min + box_width, box_x_min, box_x_min],
            [box_y_min, box_y_min, box_y_min + box_height, box_y_min + box_height, box_y_min],
            linestyle='--', linewidth=4, color='blue', alpha=0.5)

    # Calculate the centroid position of the data points for each task
    centroid_position = np.array([task_X[:, Para_Select[0]].sum() / len(task_X), task_X[:, Para_Select[1]].sum() / len(task_X)])
    box_center_x = box_x_min + box_width / 2
    box_center_y = box_y_min + box_height / 2
    task_number = (i // num_tasks_per_row) * num_tasks_per_row + (i % num_tasks_per_row) + 1
    x_range = global_x_max - global_x_min
    y_range = global_y_max - global_y_min
    circle_radius_x = 0.15 * x_range
    circle_radius_y = 0.15 * y_range
    if i in PGCL_taskcreation:
        pcolor = pcolor1
    else:
        pcolor = pcolor2
        
    circle = Ellipse((box_center_x, box_center_y), width=circle_radius_x, height=circle_radius_y, color=pcolor, fill=True, alpha=0.8)
    ax.add_patch(circle)
    ax.text(box_center_x, box_center_y, f'{task_number}', fontsize=30, color='black', weight='bold', ha='center', va='center')
    
# Add arrow lines between tasks
x_range = global_x_max - global_x_min
y_range = global_y_max - global_y_min
circle_radius_x = 0.16 * x_range/3
circle_radius_y = 0.16 * y_range/3

for i in range(len(task_inputs) - 1):
    box_center_x1 = global_x_min + (i // num_tasks_per_row) * box_width + box_width / 2
    box_center_y1 = global_y_min + (i % num_tasks_per_row) * box_height + box_height / 2
    box_center_x2 = global_x_min + ((i+1) // num_tasks_per_row) * box_width + box_width / 2
    box_center_y2 = global_y_min + ((i+1) % num_tasks_per_row) * box_height + box_height / 2
    dx = box_center_x2 - box_center_x1
    dy = box_center_y2 - box_center_y1
    start_x = box_center_x1 + circle_radius_x * dx / np.sqrt(dx**2 + dy**2)
    start_y = box_center_y1 + circle_radius_y * dy / np.sqrt(dx**2 + dy**2)
    end_x = box_center_x2 - circle_radius_x * dx / np.sqrt(dx**2 + dy**2)
    end_y = box_center_y2 - circle_radius_y * dy / np.sqrt(dx**2 + dy**2)
    ax.annotate("",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", lw=3, color="black"))

    
# Set axis labels using parameter names
ax.set_xlabel(Param_Names[Para_Select[0]], fontname='Times New Roman', fontsize=16)
ax.set_ylabel(Param_Names[Para_Select[1]], fontname='Times New Roman', fontsize=16)

# Create a custom mappable object for the colorbar using the 'autumn' colormap
from matplotlib.cm import ScalarMappable
norm = plt.Normalize(vmin=0, vmax=1)
mappable = ScalarMappable(cmap=colormap, norm=norm)
mappable.set_array([])

# Add a colorbar for the target values using the custom mappable object
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('EE')
plt.show()


# %% Plot Functions
def plot_task_error(lists_to_plot, legend_name, save_file_name, ylim=None, figsize=(4,3)):
    plt.rcParams['font.family'] = 'Times New Roman'
    line_styles = ['-', '--', '-.', ':', '--', '-']
    marker_shapes = ['o', '^', 's', 'D', 'v', 'p']
    scatter_colors = sns.color_palette("bright", len(lists_to_plot))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lst in enumerate(lists_to_plot):
        a = np.array(lst)

        if a.ndim == 3:
            b = np.nanmean(a, axis=2)
        elif a.ndim == 2:
            b = a
        else:
            raise ValueError("Invalid matrix dimension. Only 2D or 3D matrices are allowed.")

        means = np.mean(b, axis=0)
        stds = np.std(b, axis=0)
        
        ax.plot(np.arange(len(means)) + 1, means, marker=marker_shapes[i % len(marker_shapes)], markersize=8,
                linestyle=line_styles[i % len(line_styles)], mec='black', 
                mfc=scatter_colors[i % len(scatter_colors)],
                label=legend_name[i], color='black', linewidth=1,alpha=0.8)
        ax.errorbar(np.arange(len(means)) + 1, means, yerr=stds,
                    fmt='none', capsize=3, color=scatter_colors[i % len(line_styles)], zorder=1)

    ax.set_xlabel('Batch Number', fontsize=12)
    ax.set_ylabel('Seen Batch Averaged Error [%]', fontsize=12)
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
    if save_file_name:
        plt.savefig(os.path.join(subfolder_path, save_file_name), dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_task_error_1D(lists_to_plot, legend_name, save_file_name, ylim=None, figsize=(4,3)):
    plt.rcParams['font.family'] = 'Times New Roman'
    line_styles = ['-', '--', '-.', ':', '--', '-']
    marker_shapes = ['o', '^', 's', 'D', 'v', 'p']
    scatter_colors = sns.color_palette("bright", len(lists_to_plot))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lst in enumerate(lists_to_plot):
        a = np.array(lst)

        if a.ndim == 1:
            means = a
            stds = None
        elif a.ndim == 2:
            means = np.mean(a, axis=0)
            stds = np.std(a, axis=0)
        else:
            raise ValueError("Invalid matrix dimension. Only 1D or 2D matrices are allowed.")

        ax.plot(np.arange(len(means)) + 1, means, marker=marker_shapes[i % len(marker_shapes)], markersize=8,
                linestyle=line_styles[i % len(line_styles)], mec='black', 
                mfc=scatter_colors[i % len(scatter_colors)],
                label=legend_name[i], color='black', linewidth=1,alpha=0.8)
        if stds is not None:
            ax.errorbar(np.arange(len(means)) + 1, means, yerr=stds,
                        fmt='none', capsize=3, color=scatter_colors[i % len(line_styles)], zorder=1)

    ax.set_xlabel('Batch Number', fontsize=12)
    ax.set_ylabel('Seen Batch Training Time [s]', fontsize=12)
    ax.legend(fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)


    # Define the subfolder path
    subfolder_path = 'Image and Data'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if save_file_name:
        plt.savefig(os.path.join(subfolder_path, save_file_name), dpi=300, bbox_inches='tight')
    plt.show()

# %% Train EWC(CL)->EWC_SH and EWC(PGCL) ->EWC_SH_Interval_Org
RepeatTimes = 10
ewc_lambda = 0.9
accs_ewc_rep, times_ewc_rep = EWC_SH(InputLen, task_data, len(task_data), RepeatTimes, ewc_lambda,DEVICE)

accs_ewc_rep_int, times_ewc_rep_int = EWC_SH_Interval_Org(InputLen, task_data, tasks_num, RepeatTimes, ewc_lambda, DEVICE, PGCL_taskcreation)

# %% Plots Results
legend_name = ['CL (EWC)','PGCL (EWC)'] 
SaveFigName = ''
lists_to_plot = [accs_ewc_rep, accs_ewc_rep_int]
plot_task_error(lists_to_plot, legend_name, SaveFigName , ylim=None, figsize=(4,3))

legend_name = ['CL (EWC)','PGCL (EWC)'] 
SaveFigName = ''
lists_to_plot = [np.sum(np.cumsum(times_ewc_rep, axis = 1), axis = 0),np.sum(np.cumsum(times_ewc_rep_int, axis = 1), axis = 0)]
plot_task_error_1D(lists_to_plot, legend_name, SaveFigName , ylim=None, figsize=(4,3))



