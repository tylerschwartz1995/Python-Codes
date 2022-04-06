# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:22:11 2022

@author: tyler
"""

#Pytorch Gradients
import torch

#Create tensor
x = torch.tensor(2.0, requires_grad=True)
y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
print(y)

#Backpropagation in one step
y.backward()
print(x.grad)

#Backpropagation in multiple steps
#Create tensor
x = torch.tensor([[1.,2.,3.],[3,2,1]], requires_grad=True)
print(x)

#Create first layer
y = 3*x+2
print(y)

#Create Second Layer
z = 2*y**2
print(z)

#Set output to be matrix mean
out = z.mean()
print(out)

#Perform backpropagation
out.backward()
print(x.grad)















