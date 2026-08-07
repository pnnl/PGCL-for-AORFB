# -*- coding: utf-8 -*-
"""
Functions for CL and PGCL for EE Predictions
@author: Yucheng Fu
"""
from BaseFunctions import *
from DeviceSetting import *
import time
import copy
        
# %% EWC Method
def on_task_update_SH(task_id, train, model, fisher_dict, optpar_dict):
    model.train()
    criterion = MSELoss()
    device = next(model.parameters()).device
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    sample_count = 0
    for inputs, targets in train:
      model.zero_grad(set_to_none=True)
      yhat = model(inputs.to(device))
      loss = criterion(yhat, targets.to(device))
      loss.backward()
      batch_size = inputs.shape[0]
      sample_count += batch_size
      for name, param in model.named_parameters():
        if param.grad is not None:
          fisher[name] += param.grad.detach().pow(2) * batch_size
    if sample_count == 0:
      raise ValueError("Cannot estimate Fisher information from an empty task")
    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}
    for name, param in model.named_parameters():
      optpar_dict[task_id][name] = param.detach().clone()
      fisher_dict[task_id][name] = fisher[name] / sample_count

# Add the MSE regularization loss:
def train_ewc_SH(model, task_id, train, ewc_lambda, fisher_dict, optpar_dict):
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = next(model.parameters()).device
    for epoch in range(100):
      # enumerate mini batches
        for i, (inputs, targets) in enumerate(train):
            optimizer.zero_grad()
            yhat = model(inputs.to(device))
            loss = criterion(yhat, targets.to(device))
            
            #regularization term
            for task in range(task_id):
                for name, param in model.named_parameters():
                    fisher = fisher_dict[task][name]
                    optpar = optpar_dict[task][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
            loss.backward()
            optimizer.step()
    print(f"EWC_SH Task: {task_id+1}, Trained Epoch: {epoch+1} \tLoss: {loss.item():.6f}")

# Regular EWC Method
def EWC_SH(InputLen, task_data_with_overlap, tasks_num, RepeatTimes, ewc_lambda, DEVICE):
    accs_ewc_rep_SH = []
    training_times_rep = []  

    for repeat in range(0, RepeatTimes):
        model = MLP(InputLen).to(DEVICE)
        accs_ewc_SH = []
        training_times = [] 

        fisher_dict = {}
        optpar_dict = {}

        # Loop through all tasks
        for task_id in range(tasks_num):
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

            # Collect all test accuracies
            accs_ewc_SH.append(accs_subset)

        accs_ewc_rep_SH.append(accs_ewc_SH)
        training_times_rep.append(training_times)  # Add the training times for this repeat to the list

    return accs_ewc_rep_SH, training_times_rep  # Return the accuracies and training times


# EWC with PGCL which can specify the task creation with task_creation term
def EWC_SH_Interval_Org(InputLen, task_data_with_overlap, tasks_num, RepeatTimes, ewc_lambda, DEVICE, task_creation):
    accs_ewc_rep_SH = []
    training_times_rep = []

    for repeat in range(0, RepeatTimes):
        model = MLP(InputLen).to(DEVICE)
        accs_ewc_SH = []
        training_times = []

        # Define dictionaries to store values needed by EWC
        fisher_dict = {}
        optpar_dict = {}
        head_idx = -1
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]
            # Determine the head index to use for the current task
            if task_id in task_creation:
                # Determine the head index to use for the current task
                head_idx = head_idx + 1
                # Train the model (with the new head) on the current task
                start_time = time.time()
                train_ewc_SH(model,  head_idx, train, ewc_lambda, fisher_dict, optpar_dict)
                on_task_update_SH( head_idx, train, model, fisher_dict, optpar_dict)
                training_time = time.time() - start_time
                training_times.append(training_time)
            else:
                training_times.append(0)
            # Test the model on all tasks seen so far
            accs_subset = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, model)
                accs_subset.append(mse)

            # For unseen tasks, we don't test
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))

            # Collect all test accuracies
            accs_ewc_SH.append(accs_subset)

        accs_ewc_rep_SH.append(accs_ewc_SH)
        training_times_rep.append(training_times)
    return accs_ewc_rep_SH, training_times_rep


# EWC with PGCL that return the extrat_testing results
def EWC_SH_Interval(InputLen, task_data_with_overlap, tasks_num, RepeatTimes, ewc_lambda, DEVICE, extra_testing):
    accs_ewc_rep_SH = []
    accs_ewc_rep_SH_ex = []
    accs_ews_rep_SH_ex_pre = []
    training_times_rep = []

    for repeat in range(0, RepeatTimes):
        model = MLP(InputLen).to(DEVICE)
        accs_ewc_SH = []
        accs_ewc_SH_ex = []
        accs_ews_SH_ex_pre = []
        training_times = []

        fisher_dict = {}
        optpar_dict = {}

        # Loop through all tasks
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]

            # Determine the head index to use for the current task
            if task_id in [0, 3, 6]:
                # Determine the head index to use for the current task
                if task_id in [0, 1, 2]:
                    head_idx = 0
                elif task_id in [3, 4, 5]:
                    head_idx = 1
                else:
                    head_idx = 2
                # Train the model (with the new head) on the current task
                start_time = time.time()
                train_ewc_SH(model,  head_idx, train, ewc_lambda, fisher_dict, optpar_dict)
                on_task_update_SH( head_idx, train, model, fisher_dict, optpar_dict)
                training_time = time.time() - start_time
                training_times.append(training_time)
            else:
                training_times.append(0)


            # Test the model on all tasks seen so far
            accs_subset = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, model)
                accs_subset.append(mse)
            # For unseen tasks, we don't test
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))
            # Collect all test accuracies
            accs_ewc_SH.append(accs_subset)
            _, predictions_ex, actuals_ex = evaluate_model(extra_testing, model)
            accs_ewc_SH_ex.append(abs((predictions_ex-actuals_ex)/actuals_ex).tolist())
            accs_ews_SH_ex_pre.append(predictions_ex.tolist())
        accs_ewc_rep_SH.append(accs_ewc_SH)
        accs_ewc_rep_SH_ex.append(accs_ewc_SH_ex)
        accs_ews_rep_SH_ex_pre.append(accs_ews_SH_ex_pre)
        training_times_rep.append(training_times)

    return accs_ewc_rep_SH, accs_ewc_rep_SH_ex, accs_ews_rep_SH_ex_pre, training_times_rep


# %% LwF method for continual regression with one shared output head
def train_LwF(train, model, task_id, LwF_lambda, teacher=None):
    criterion = MSELoss()
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    if teacher is not None:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    for epoch in range(500):
        for inputs, targets in train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            predictions = model(inputs)
            loss_new = criterion(predictions, targets)
            if teacher is None:
                loss = loss_new
            else:
                with torch.no_grad():
                    old_predictions = teacher(inputs)
                loss_old = criterion(predictions, old_predictions)
                loss = loss_new + LwF_lambda * loss_old

            loss.backward()
            optimizer.step()
    print(f"LwF Task: {task_id+1}, Trained Epoch: {epoch+1} \tLoss: {loss.item():.6f}")
        

def LwF(InputLen, task_data_with_overlap, tasks_num, LwF_lambda, RepeatTimes, DEVICE):
    accs_LwF_rep = []
    training_times_rep = []

    for repeat in range(0, RepeatTimes):
        model = MLP(InputLen).to(DEVICE)
        accs_LwF = []
        training_times = []  

        # Loop through all tasks
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]
            
            teacher = copy.deepcopy(model) if task_id > 0 else None
            start_time = time.time()
            train_LwF(train, model, task_id, LwF_lambda, teacher)
            training_time = time.time() - start_time 
            training_times.append(training_time) 

            # Test the model on all tasks seen so far
            accs_subset = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, model)
                accs_subset.append(mse)
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num-1 - task_id))
            # Collect all test accuracies
            accs_LwF.append(accs_subset)

        accs_LwF_rep.append(accs_LwF)
        training_times_rep.append(training_times)  # Add the training times for this repeat to the list

    return accs_LwF_rep, training_times_rep  # Return the accuracies and training times
