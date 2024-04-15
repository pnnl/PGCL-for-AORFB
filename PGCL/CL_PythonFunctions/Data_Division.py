# -*- coding: utf-8 -*-
"""
Provide Functions to Divide the ASO Dataset according to different training strategies
@author:Yucheng Fu
"""
import numpy as np
from BaseFunctions import *
from DeviceSetting import *
# %%

def task_division(X, y, method, tasks_num, Para_Select=None):
    if method == 'Input':
        if Para_Select is None:
            raise ValueError('A single variable index must be provided for method "Input"')
        task_data = Division_by_Input(X, y, tasks_num, Para_Select)
    elif method == 'Output':
        task_data = Division_by_Output(X, y, tasks_num)
    elif method == 'Random':
        task_data = Division_Random(X, y, tasks_num)
    elif method == 'TwoInputs':
        if Para_Select is None or len(Para_Select) != 2:
            raise ValueError('Two variable indices must be provided for method "TwoInputs"')
        task_data = Division_by_TwoInputs(X, y, tasks_num, Para_Select)
    else:
        raise ValueError('Invalid method specified')
    
    return task_data

def Division_by_TwoInputs(X, y, tasks_num, Para_Select):
    # Calculate the number of tasks along each input dimension
    tasks_dim = int(np.sqrt(tasks_num))

    # Check if the total number of tasks is a perfect square
    if tasks_dim * tasks_dim != tasks_num:
        raise ValueError('tasks_num must be a perfect square for method "TwoInputs"')

    # Calculate the task bounds for each input parameter
    task_bounds = []
    for idx in Para_Select:
        task_lowbound = np.min(X[:, idx])
        task_upbound = np.max(X[:, idx])
        task_interval = (task_upbound - task_lowbound) / tasks_dim
        task_bounds.append([(task_lowbound + i * task_interval, task_lowbound + (i + 1) * task_interval) for i in range(tasks_dim)])

    task_data = []
    task_id = 0
    for i in range(tasks_dim):
        for j in range(tasks_dim):
            mask = np.where((X[:, Para_Select[0]] >= task_bounds[0][i][0]) &
                            (X[:, Para_Select[0]] < task_bounds[0][i][1]) &
                            (X[:, Para_Select[1]] >= task_bounds[1][j][0]) &
                            (X[:, Para_Select[1]] < task_bounds[1][j][1]))
            tempX = X[mask, :].squeeze()
            tempy = y[mask]
            train, test = prepare_data(tempX, tempy)
            task_data.append((train, test))
            task_id += 1

    return task_data


def Division_by_Input(X,y, tasks_num, Para_Select):
    # Implement method A using var
    task_lowbound = np.min(X[:,Para_Select]) 
    task_upbound = np.max(X[:,Para_Select])
    task_interval = (task_upbound-task_lowbound)/tasks_num
    task_classes_arr = [None]*tasks_num
    task_data = []
    for task_id in range(tasks_num):
        task_classes_arr[task_id] = (task_lowbound+task_id*task_interval, task_lowbound+(task_id+1)*task_interval )
        mask = np.where((X[:,Para_Select] >= task_classes_arr[task_id][0] )&
                        (X[:,Para_Select] < task_classes_arr[task_id][1]) ) 
        print(len(mask[0]))
        tempX = X[mask,:].squeeze()
        tempy = y[mask]
        train, test = prepare_data(tempX, tempy)
        print(len(train))
        task_data.append((train,test))
    return task_data

def Division_by_Output(X,y, tasks_num):
    # Implement method B
    task_lowbound = np.min(y) 
    task_upbound = np.max(y)
    task_interval = (task_upbound-task_lowbound)/tasks_num
    task_classes_arr = [None]*tasks_num
    task_data = []
    for task_id in range(tasks_num):
        task_classes_arr[task_id] = (task_lowbound+task_id*task_interval, task_lowbound+(task_id+1)*task_interval )
        mask = np.where((y >= task_classes_arr[task_id][0] )&
                        (y < task_classes_arr[task_id][1]) ) 
        print(len(mask[0]))
        tempX = X[mask,:].squeeze()
        tempy = y[mask]
        train, test = prepare_data(tempX, tempy)
        print(len(train))
        task_data.append((train,test))  
    return task_data

def Division_Random(X,y, tasks_num):
    task_data = []
    a = np.arange(0,X.shape[0],1)
    np.random.shuffle(a)
    X = X[a,:]
    y = y[a]
    for task_id in range(tasks_num):
        mask = np.arange(task_id*700,(task_id+1)*700)
        print(len(mask))
        tempX = X[mask,:].squeeze()
        tempy = y[mask]
        train, test = prepare_data(tempX, tempy)
        print(len(train))
        task_data.append((train,test))
    return task_data

