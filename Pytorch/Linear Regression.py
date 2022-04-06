# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:33:56 2022

@author: tyler
"""

#Import
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

#Create column matrix of x values
X = torch.linspace(1,50,50).reshape(-1,1)
print(X)

#Create a random array of error values
torch.manual_seed(71)
e = torch.randint(-8,9,(50,1),dtype=torch.float)
print(e.sum())
print(e)

#Create a matrix of y values
y = 2*X + 1+ e
print(y.shape)

#Plot results
plt.scatter(X.numpy(), y.numpy())
plt.ylabel('y')
plt.xlabel('x');

#Simple linear model
torch.manual_seed(59)
model = nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)

#Model classes
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
#Callin created model class
torch.manual_seed(59)
model = Model(1,1)
print(model)
print('Weight:', model.linear.weight.item())
print('Bias:  ', model.linear.bias.item())

#Alternate call
for name, param in model.named_parameters():
    print(name, '\t', param.item())

#Pass tensor into model class forward function
x = torch.tensor([2.0])
print(model.forward(x))

#Plot the initial model
x1 = np.array([X.min(), X.max()])
print(x1)

w1, b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Initial weight: {w1:.8f}, Initial bias: {b1:.8f}')

y1 = x1*w1 + b1
print(y1)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.title('Initial Model')
plt.ylabel('y')
plt.xlabel('x');

#Set loss function
criterion = nn.MSELoss()

#Set optimization (SGD with learning rate = 0.001)
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

#Training model (epoch is a single pass through entire dataset)
#Batch is each gradient update

epochs = 50
losses = []

for i in range(epochs):
    i+=1
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}  weight: {model.linear.weight.item():10.8f}  \
          bias: {model.linear.bias.item():10.8f}') 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() #Update
    
#Plot loss values
for i in range(0,50):
    losses[i] = losses[i].detach().numpy()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

#Plot result
w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Current weight: {w1:.8f}, Current bias: {b1:.8f}')
print()

y1 = x1*w1 + b1
print(x1)
print(y1)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.title('Current Model')
plt.ylabel('y')
plt.xlabel('x');


























