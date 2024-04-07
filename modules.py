import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pdToDataLoader(train_df, valid_df, test_df, target_column, batch_size):
    numeric_train_df = train_df.drop(columns=[target_column, 'Date'])
    numeric_valid_df = valid_df.drop(columns=[target_column, 'Date'])
    numeric_test_df = test_df.drop(columns=[target_column, 'Date'])

    X_train = numeric_train_df.values.astype(np.float32)
    y_train = train_df[target_column].values
    X_valid = numeric_valid_df.values.astype(np.float32)
    y_valid = valid_df[target_column].values
    X_test = numeric_test_df.values.astype(np.float32)
    y_test = test_df[target_column].values

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_valid_tensor = torch.tensor(X_valid)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

