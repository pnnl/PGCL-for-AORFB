# -*- coding: utf-8 -*-
"""
Functions for CL and PGCL for EE Predictions
@author: Yucheng Fu
"""
from BaseFunctions import *
from DeviceSetting import *
import time
        
# %% EWC Method
def on_task_update_SH(task_id, train, model, fisher_dict, optpar_dict):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    criterion = MSELoss()

    for i, (inputs, targets) in enumerate(train):
      yhat = model(inputs)
      loss = criterion(yhat, targets)
      loss.backward()
  
    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}
    for name, param in model.named_parameters():
      optpar_dict[task_id][name] = param.data.clone()
      fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

# Add the MSE regularization loss:
def train_ewc_SH(model, task_id, train, ewc_lambda, fisher_dict, optpar_dict):
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
      # enumerate mini batches
        for i, (inputs, targets) in enumerate(train):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            
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
        model = MLP(InputLen)
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
        model = MLP(InputLen)
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
        model = MLP(InputLen)
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


# %% LwF method
def train_LwF(train, model, task_id,LwF_lambda):
    criterion = MSELoss()
    optimizer = []
    for task in range(task_id+1):
        tempopt = optim.Adam(model[task].parameters(), lr=0.01) 
        optimizer.append( tempopt)
    for epoch in range(500):
        for i, (inputs, targets) in enumerate(train):
          for task in range(task_id+1):
              optimizer[task].zero_grad()
          loss_old = torch.tensor(0.0)
          for task in range(task_id):
              targets_old = model[task](inputs)
              loss_old += loss_old + criterion(targets, targets_old)
              
          targets_new = model[task_id](inputs)
          loss_new = criterion(targets, targets_new)   
          if task_id > 0:
              loss = LwF_lambda*loss_old + loss_new
          else:
              loss = loss_new
          loss.backward()
          for task in range(task_id+1):
              optimizer[task].step()
    print(f"LwF Task: {task_id+1}, Trained Epoch: {epoch+1} \tLoss: {loss.item():.6f}")
        

def LwF(InputLen, task_data_with_overlap, tasks_num, LwF_lambda, RepeatTimes, DEVICE):
    accs_LwF_rep = []
    accs_LwF_rep_v2 = []
    training_times_rep = []

    for repeat in range(0, RepeatTimes):
        base = MLP_MH(InputLen)
        heads = []
        accs_LwF = []
        accs_LwF_v2 = []
        training_times = []  

        # Loop through all tasks
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]
            
            # Define a new head for this task
            model = FHeadNet(base).to(DEVICE)
            heads.append(model)
            
            start_time = time.time()  
            train_LwF(train, heads, task_id, LwF_lambda)
            training_time = time.time() - start_time 
            training_times.append(training_time) 

            # Test the model on all tasks seen so far
            accs_subset = []
            accs_subset_v2 = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]
                mse, predictions, actuals = evaluate_model(test, heads[i])
                accs_subset.append(mse)
                mse2, predictions2, actuals2 = evaluate_model(test, heads[task_id])
                accs_subset_v2.append(mse2)
                # For unseen tasks, we don't test
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num-1 - task_id))
                accs_subset_v2.extend([np.nan] * (tasks_num-1 - task_id))
            # Collect all test accuracies
            accs_LwF.append(accs_subset)
            accs_LwF_v2.append(accs_subset_v2)

        accs_LwF_rep.append(accs_LwF)
        accs_LwF_rep_v2.append(accs_LwF_v2)
        training_times_rep.append(training_times)  # Add the training times for this repeat to the list

    return accs_LwF_rep, training_times_rep  # Return the accuracies and training times


# LwF for PGCL with selective task training, the task can be assigned similar to EWC Method if adapted for different applications
def LwF_Interval(InputLen, task_data_with_overlap, tasks_num, LwF_lambda, RepeatTimes, DEVICE):
    accs_LwF_rep = []
    accs_LwF_rep_v2 = []
    training_times_rep = []  # New list to store training times

    for repeat in range(0, RepeatTimes):
        base = MLP_MH(InputLen)
        heads = []
        accs_LwF = []
        accs_LwF_v2 = []
        training_times = []  # New list to store training times for each task

        # Loop through all tasks
        for task_id in range(tasks_num):
            # Collect the training data for the new task
            train, test = task_data_with_overlap[task_id]

            # Create a new head for tasks 1, 4, and 7
            if task_id in [0, 3, 6]:
                model = FHeadNet(base).to(DEVICE)
                heads.append(model)
                start_time = time.time()

                # Determine the head index to use for the current task
                if task_id in [0, 1, 2]:
                    head_idx = 0
                elif task_id in [3, 4, 5]:
                    head_idx = 1
                else:
                    head_idx = 2

                train_LwF(train, heads, head_idx, LwF_lambda)
                training_time = time.time() - start_time

            else:
                training_time = 0  # Use the same training time as the previous task

            training_times.append(training_time)  # Add training time to the list

            # Test the model on all tasks seen so far
            accs_subset = []
            accs_subset_v2 = []
            for i in range(0, task_id + 1):
                _, test = task_data_with_overlap[i]

                # Determine the head index to use for the testing task
                if i in [0, 1, 2]:
                    test_head_idx = 0
                elif i in [3, 4, 5]:
                    test_head_idx = 1
                else:
                    test_head_idx = 2

                mse, predictions, actuals = evaluate_model(test, heads[test_head_idx])
                accs_subset.append(mse)
                mse2, predictions2, actuals2 = evaluate_model(test, heads[test_head_idx])
                accs_subset_v2.append(mse2)

            # For unseen tasks, we don't test
            if task_id < (tasks_num - 1):
                accs_subset.extend([np.nan] * (tasks_num - 1 - task_id))
                accs_subset_v2.extend([np.nan] * (tasks_num - 1 - task_id))

            # Collect all test accuracies
            accs_LwF.append(accs_subset)
            accs_LwF_v2.append(accs_subset_v2)

        accs_LwF_rep.append(accs_LwF)
        accs_LwF_rep_v2.append(accs_LwF_v2)
        training_times_rep.append(training_times)  # Add the training times for this repeat to the list

    return accs_LwF_rep, training_times_rep
