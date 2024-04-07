from modules import *
from models.baseline import Baseline
import pandas as pd
import torch.optim as optim
import torch.nn as nn

PATH = 'datasets/'
train_df = pd.read_csv(PATH + 'train_dataset.csv')
valid_df = pd.read_csv(PATH + 'valid_dataset.csv')
test_df = pd.read_csv(PATH + 'test_dataset.csv')

train_dataloader, valid_dataloader, test_dataloader = pdToDataLoader(
    train_df, 
    valid_df, 
    test_df, 
    'target', 
    32
)

input_size = 14
hidden_size = 32
num_layers = 2
num_classes = 2

# models
baseline_model = Baseline(
    input_size,
    hidden_size,
    num_layers,
    num_classes
)

# loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    
# training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    baseline_model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch in train_dataloader:
        
        x_batch, y_batch = torch.permute(batch[0].unsqueeze(0), (1, 0, 2)), batch[1]

        y_logits = baseline_model(x_batch)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        
        loss = loss_function(y_logits, y_batch)
        acc = accuracy(y_batch, y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += acc
    
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    
    print(f"EPOCH: {epoch+1}/{EPOCHS} | training loss: {train_loss:.4f} | training acc: {train_acc:.2f}")
    