# -*- coding: utf-8 -*-
"""
@author: Yucheng Fu
Testing the PGCL capability for predicting unseen ASO materials. Using the DHP isomer as an example
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

#%% Prepare DHP Data (Output EE from ID 780 cm2 Cell)
Phenazine_data = [
    [0.5, 59600, 0.6295, 1.47E-04, 1E-5, 425, 1.64E-09, 0.0065, -0.88, 50, 0.7197],
    [0.5, 59600, 0.6295, 1.47E-04, 1E-5, 373.8, 5.2E-10, 0.0065, -0.8, 50, 0.6964],
    [0.5, 59600, 0.6295, 1.47E-04, 1E-5, 324, 2.1E-09, 0.0065, -0.98, 50, 0.7238],
    [0.5, 59600, 0.6295, 1.47E-04, 1E-5, 552, 1.2E-09, 0.0065, -1.08, 50, 0.7662],
    [0.5, 59600, 0.6295, 1.47E-04, 1E-5, 152, 1.2E-10, 0.0065, -0.88, 50, 0.5222]
]

Phenazine_test_X = np.array(Phenazine_data)[:, :-1].astype(np.float32)
Phenazine_test_y = np.array(Phenazine_data)[:, -1].astype(np.float32)
Phenazine_test_X = min_max_scaler.transform(Phenazine_test_X)

Phenazine_test = prepare_data_test(Phenazine_test_X,Phenazine_test_y)

column_ranges = []
for column in range(Phenazine_test_X.shape[1]):
    min_value = Phenazine_test_X[:, column].min()
    max_value = Phenazine_test_X[:, column].max()
    column_ranges.append((min_value, max_value))

print("Column Ranges:")
for column, (min_value, max_value) in enumerate(column_ranges):
    print(f"Column {column + 1}: Min={min_value:.4f}, Max={max_value:.4f}")

column_ranges = []
for column in range(X.shape[1]):
    min_value = X[:, column].min()
    max_value = X[:, column].max()
    column_ranges.append((min_value, max_value))

print("Column Ranges:")
for column, (min_value, max_value) in enumerate(column_ranges):
    print(f"Column {column + 1}: Min={min_value:.4f}, Max={max_value:.4f}")
# %% Task Division

set_seed(seed=SEED)

# Divide task by two inputs
tasks_num = 9
Division_method = 'TwoInputs'
Para_Select = [8,5]; #  E_n  and c_n
task_data = task_division(X,y, Division_method, tasks_num,Para_Select)

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



# %% Data Batch Visualization
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
    sc = ax.scatter(task_X[:, Para_Select[0]], task_X[:, Para_Select[1]], alpha=0.2, c='gray', edgecolors='black', s=80)
    box_x_min = global_x_min + (i // num_tasks_per_row) * box_width
    box_y_min = global_y_min + (i % num_tasks_per_row) * box_height
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
    if i in [0, 3,6]:
        pcolor = pcolor1
    else:
        pcolor = pcolor2
    

# Get the single task data from task_inputs_ex
task_X_ex = task_inputs_ex[0]
task_y_ex = task_targets_ex[0]
sc = ax.scatter(task_X_ex[:, Para_Select[0]], task_X_ex[:, Para_Select[1]], alpha=1, c='red', edgecolors='black', s=80)
ax.set_xlabel(Param_Names[Para_Select[0]], fontname='Times New Roman', fontsize=20)
ax.set_ylabel(Param_Names[Para_Select[1]], fontname='Times New Roman', fontsize=20)

# Create a custom mappable object for the colorbar using the 'autumn' colormap
from matplotlib.cm import ScalarMappable
norm = plt.Normalize(vmin=0, vmax=1)
mappable = ScalarMappable(cmap=colormap, norm=norm)
mappable.set_array([])

# Add a colorbar for the target values using the custom mappable object
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('EE',fontsize=20)
cbar.ax.tick_params(labelsize=20)  # Adjust the size as needed
plt.show()


# %% Define Plot Functions
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

    ax.set_xlabel('Batch Number', fontsize=14)
    ax.set_ylabel('Seen Batch Training Time [s]', fontsize=14)
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

# %% Train PGCL EWC Model
ewc_lambda = 0.9
accs_ewc_rep_int, accs_ewc_rep_SH_ex, accs_ewc_rep_SH_ex_pre, times_ewc_rep_int = EWC_SH_Interval(InputLen, task_data, len(task_data), RepeatTimes, ewc_lambda,DEVICE,Phenazine_test)

# %% Error plot for different number of training Tasks

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
line_styles = ['-', '--', '-.', ':', '--', '-']
marker_shapes = ['o', '^', 's', 'D', 'v', 'p']

a = np.nanmean(accs_ewc_rep_SH_ex, axis=0).squeeze()
row_indices = [0, 3, 6]

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Loop through and plot each selected row with different colors
for i, row_index in enumerate(row_indices):
    color = plt.cm.jet(i / len(row_indices))  # Get a unique color for each line
    label_text = f'PGCL using training data batch {row_index + 1}' if row_index + 1 == 1 else f'PGCL with training data batch 1-{row_index + 1}'
    
    ax.plot(range(1, a.shape[1] + 1), a[row_index, :] * 100, label=label_text,
             marker=marker_shapes[i % len(marker_shapes)], markersize=12, linestyle=line_styles[i % len(line_styles)],
             mec='black', mfc=color, linewidth=1, alpha=0.7, color=color)

# Customize the plot
ax.set_xlabel('DHP Isomers')
ax.set_ylabel('EE Error[%]')
ax.set_xticks(range(1, a.shape[1] + 1))
ax.set_xticklabels(['1,3-DHP', '1,4-DHP', '1,6-DHP', '1,8-DHP', '1,9-DHP'])


ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
ax.tick_params(direction='in', length=8, width=2, which='major')
ax.tick_params(direction='in', length=8, width=1, which='minor')
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.grid(True, which='both', linestyle='-', linewidth=0.7)
ax.legend()
plt.show()


# %% Averaged Error for all DHP Isomers
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24  # You can adjust this value as needed
average_error = np.mean(a, axis=1)
training_indices = [0, 3, 6]

fig, ax = plt.subplots(figsize=(8, 6))
# Plot the average error
ax.plot(range(1, a.shape[0] + 1), average_error*100, label='Averaged for all DHP Isomers', 
         marker='o', markersize=8, linestyle='--', linewidth=2,
         mec='black')
# Customize the plot
plt.xlabel('Training Data Batch Number')
plt.ylabel('EE Error [%]')
ax.tick_params(top=True, right=True, direction='in', length=6, width=2, which='both')
ax.tick_params(direction='in', length=8, width=2, which='major')
ax.tick_params(direction='in', length=8, width=1, which='minor')
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.grid(True, which='both', linestyle='-', linewidth=0.7)
plt.legend()
plt.ylim([5,25])
plt.grid(True)
plt.show()


