import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from tqdm.auto import tqdm

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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    X_valid_tensor = torch.tensor(X_valid)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

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

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch in dataloader:
        
        x_batch, y_batch = torch.permute(batch[0].unsqueeze(0), (1, 0, 2)), batch[1]

        y_logits = model(x_batch).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        
        loss = loss_fn(y_logits, y_batch)
        acc = accuracy(y_batch, y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += acc
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch in dataloader:
            x_batch, y_batch = torch.permute(batch[0].unsqueeze(0), (1, 0, 2)), batch[1]

            y_logits = model(x_batch).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            
            if y_logits.dim() == 0:
                y_logits = y_logits.unsqueeze(0)
                y_pred = y_pred.unsqueeze(0)
                
            loss = loss_fn(y_logits, y_batch)
            acc = accuracy(y_batch, y_pred)
            
            test_loss += loss.item()
            test_acc += acc
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          valid_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        valid_loss, valid_acc = test_step(model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

    # 6. Return the filled results at the end of the epochs
    return results