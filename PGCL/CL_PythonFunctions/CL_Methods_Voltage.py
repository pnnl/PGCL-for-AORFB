# -*- coding: utf-8 -*-
"""
Functions for CL and PGCL for Voltage Predictions
@author: Yucheng Fu
"""
from BaseFunctions import *
from DeviceSetting import *
import time

# DNN naive CL Implementation
def CNN(InputLen, task_data, tasks_num, RepeatTimes, extra_testing=None):
    accs_naive_rep = []
    # Initialize these lists only if extra_testing is provided
    if extra_testing is not None:
        accs_naive_rep_ex = []
        accs_naive_rep_ex_pre = []

    for repeat in range(RepeatTimes):
        model = MLP(InputLen)
        accs_naive = []

        # These lists are used only if extra_testing is provided
        if extra_testing is not None:
            accs_naive_ex = []
            accs_naive_ex_pre = []

        for task_id in range(tasks_num):
            train, test = task_data[task_id]
            start_time = time.time()
            train_model(train, model)
            training_time = time.time() - start_time

            accs_subset = []
            for i in range(task_id + 1):
                _, test = task_data[i]
                mse, predictions, actuals = evaluate_model(test, model)
                accs_subset.append(mse)

            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))

            # Handle extra_testing if provided
            if extra_testing is not None:
                _, predictions_ex, actuals_ex = evaluate_model(extra_testing, model)
                accs_naive_ex.append(abs((predictions_ex - actuals_ex) / actuals_ex).tolist())
                accs_naive_ex_pre.append(predictions_ex.tolist())

            accs_naive.append(accs_subset)

            print(f'Repeat of {repeat + 1}, Task of {task_id + 1}, Training data size: {len(train)}, Training time: {training_time:.2f} seconds')

        accs_naive_rep.append(accs_naive)
        # Only append these if extra_testing was provided
        if extra_testing is not None:
            accs_naive_rep_ex.append(accs_naive_ex)
            accs_naive_rep_ex_pre.append(accs_naive_ex_pre)

    # Return the appropriate lists based on whether extra_testing was provided
    if extra_testing is not None:
        return accs_naive_rep, accs_naive_rep_ex, accs_naive_rep_ex_pre
    else:
        return accs_naive_rep

# %% EWC Method
def on_task_update_SH(task_id, train, model, fisher_dict, optpar_dict):
   # define the optimization
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    # define the optimization
    criterion = MSELoss()
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train):
      # compute the model output
      yhat = model(inputs)
      # calculate loss
      loss = criterion(yhat, targets)
      # credit assignment
      loss.backward()
  
    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}
    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
      optpar_dict[task_id][name] = param.data.clone()
      fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

# We also need to modify our train function to add the new regularization loss:
def train_ewc_SH(model, task_id, train, ewc_lambda, fisher_dict, optpar_dict):

    # define the optimization
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # enumerate epochs
    for epoch in range(100):
      # enumerate mini batches
        for i, (inputs, targets) in enumerate(train):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            
            ### magic here! :-)
            for task in range(task_id):
                for name, param in model.named_parameters():
                    fisher = fisher_dict[task][name]
                    optpar = optpar_dict[task][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
            loss.backward()
            optimizer.step()
    print(f"EWC_SH Task: {task_id+1}, Trained Epoch: {epoch+1} \tLoss: {loss.item():.6f}")

def EWC_SH(InputLen, task_data_with_overlap, tasks_num, RepeatTimes, ewc_lambda, DEVICE, extra_testing = None):
    accs_ewc_rep_SH = []
    training_times_rep = []  # New list to store training times
    
    if extra_testing is not None:
        accs_ewc_rep_ex = []
        accs_ewc_rep_ex_pre = []
        
        
    for repeat in range(0, RepeatTimes):
        model = MLP(InputLen)
        accs_ewc_SH = []
        training_times = []  # New list to store training times for each task

        # Define dictionaries to store values needed by EWC
        fisher_dict = {}
        optpar_dict = {}
        
        if extra_testing is not None:
            accs_ewc_ex = []
            accs_ewc_ex_pre = []
        # Loop through all tasks
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]

            # Train the model (with the new head) on the current task
            start_time = time.time()
            train_ewc_SH(model, task_id, train, ewc_lambda, fisher_dict, optpar_dict)
            on_task_update_SH(task_id, train, model, fisher_dict, optpar_dict)
            training_time = time.time() - start_time
            training_times.append(training_time)


            # Test the model on all tasks seen so far
            accs_subset = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, model)
                accs_subset.append(mse)

            # For unseen tasks, we don't test
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))

            # Handle extra_testing if provided
            if extra_testing is not None:
                _, predictions_ex, actuals_ex = evaluate_model(extra_testing, model)
                accs_ewc_ex.append(abs((predictions_ex - actuals_ex) / actuals_ex).tolist())
                accs_ewc_ex_pre.append(predictions_ex.tolist())

            # Collect all test accuracies
            accs_ewc_SH.append(accs_subset)

        accs_ewc_rep_SH.append(accs_ewc_SH)
        training_times_rep.append(training_times)  # Add the training times for this repeat to the list
        # Only append these if extra_testing was provided
        if extra_testing is not None:
            accs_ewc_rep_ex.append(accs_ewc_ex)
            accs_ewc_rep_ex_pre.append(accs_ewc_ex_pre)

    if extra_testing is not None:
        return accs_ewc_rep_SH, accs_ewc_rep_ex, accs_ewc_rep_ex_pre, training_times_rep 
    else:
        return accs_ewc_rep_SH, training_times_rep  


# %% LwF
import torch
import torch.optim as optim
from torch.nn import MSELoss
import time
import numpy as np

def train_LwF(train, model, task_id, LwF_lambda, lr, num_epochs):
    criterion = MSELoss()
    optimizer = []
    for task in range(task_id + 1):
        tempopt = optim.Adam(model[task].parameters(), lr=lr)
        optimizer.append(tempopt)
    
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train):
            # clear the gradients
            for task in range(task_id + 1):
                optimizer[task].zero_grad()
            
            loss_old = torch.tensor(0.0)
            for task in range(task_id):
                targets_old = model[task](inputs)
                loss_old += criterion(targets, targets_old)
            
            targets_new = model[task_id](inputs)
            loss_new = criterion(targets, targets_new)
            if task_id > 0:
                loss = LwF_lambda * loss_old + loss_new
            else:
                loss = loss_new
            
            loss.backward()
            for task in range(task_id + 1):
                optimizer[task].step()
    
    print(f"LwF Task: {task_id + 1}, Trained Epoch: {epoch + 1} \tLoss: {loss.item():.6f}")

def LwF_V(InputLen, task_data_with_overlap, tasks_num, LwF_lambda, RepeatTimes, DEVICE, lr, num_epochs):
    accs_LwF_rep = []
    accs_LwF_rep_v2 = []
    training_times_rep = []

    for repeat in range(RepeatTimes):
        base = MLP_MH(InputLen)  # Assuming this is defined elsewhere
        heads = []
        accs_LwF = []
        accs_LwF_v2 = []
        training_times = []

        for task_id in range(tasks_num):
            train, test = task_data_with_overlap[task_id]
            model = FHeadNet(base).to(DEVICE)  # Assuming FHeadNet is defined elsewhere
            heads.append(model)
            
            start_time = time.time()
            train_LwF(train, heads, task_id, LwF_lambda, lr, num_epochs)
            training_time = time.time() - start_time
            training_times.append(training_time)

            accs_subset = []
            accs_subset_v2 = []
            for i in range(task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, heads[i])  # Assuming evaluate_model is defined elsewhere
                accs_subset.append(mse)
                mse2, predictions2, actuals2 = evaluate_model(test, heads[task_id])
                accs_subset_v2.append(mse2)
            
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))
                accs_subset_v2.extend([np.nan] * (tasks_num - 1 - task_id))
            
            accs_LwF.append(accs_subset)
            accs_LwF_v2.append(accs_subset_v2)

        accs_LwF_rep.append(accs_LwF)
        accs_LwF_rep_v2.append(accs_LwF_v2)
        training_times_rep.append(training_times)

    return accs_LwF_rep, training_times_rep
