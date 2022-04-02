# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:18:55 2022

@author: tyler
"""

#Import libraries
import torch
import numpy as np

#See verison of pytorch
torch.__version__

#Convert numpy arrays to pytorch tensors

#Single dimension
arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))

x = torch.from_numpy(arr) #Sharing memory (changing one changes the other)
print(x)
print(type(x))
print(x.type())

#Multiple dimensions
arr2 = np.arange(0.,12.).reshape(4,3)
print(arr2)

x2 = torch.from_numpy(arr2)
print(x2)
print(type(x2))

#Copying memory
arr = np.arange(0,5)
t = torch.tensor(arr) #Make copy
print(t)

arr[2] = 77
print(t)
print(arr)

#Class constructors (torch.Tensor changes automatically to float)
data = np.array([1,2,3])

a = torch.Tensor(data)
print(a, a.type())

b = torch.tensor(data)
print(b, b.type())

c = torch.tensor(data, dtype = torch.long)
print(c, c.type())

#Creating tensors from scratch
#Empty tensors
x = torch.empty(4,3)
print(x)

#Zeros
x = torch.zeros(4, 3, dtype=torch.int64)
print(x)

#Ones
x = torch.ones(4,3)
print(x)

#Tensors from ranges
x = torch.arange(0,18,2).reshape(3,3)
print(x)

x = torch.linspace(0,18,12).reshape(3,4)
print(x)

#Tensors from data
x = torch.tensor([1,2,3,4])
print(x)
print(x.dtype)
print(x.type())

#Changing dytpe of existing tensors
print('Old:', x.type())
x = x.type(torch.int64)
print('New:', x.type())

#Random number tensors (torch.rand, torch.randn, torch.randint)
x = torch.rand(4,3) #Between 0 and 1
print(x)

x - torch.randn(4,3) #Normal distribution
print(x)

x = torch.randint(0,6, (4,3))
print(x)

#Random number tensors that follow the input size
x = torch.zeros(2,5)
print(x)

x2 = torch.randn_like(x)
print(x2)

x3 = torch.ones_like(x2)
print(x3)

#Setting random seed
torch.manual_seed(1234)
x = torch.rand(2,3)
print(x)

#Tensor attributes
x.shape #also can use x.size()

x.device #Check if cpu or gpu is being used
#Operations between tensors can only happen for tensors installed on the same device

x.layout

##Operations##

#Indexing and slicing
x = torch.arange(6).reshape(3,2)
print(x)

x[:,1] # Grabbing the right hand column values
x[:,1:3] # Grabbing the right hand column as a (3,1) slice

#Reshape tensors with .view()
x = torch.arange(10)
print(x)

x.view(2,5)
x.view(5,2)

print(x) #Unchanged

z = x.view(2,5)
print(z) #Changed

#View can infer corrrect size by placing -1
x.view(5,-1)
x.view(-1,2)

#Adopt another tensors shape
x.view_as(z)

#Tensor arithmetic
a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype = torch.float)
print(a+b)
#or
print(torch.add(a,b))

#Ouput
result = torch.empty(3)
torch.add(a,b, out = result)
print(result)

#Inplace
a.add_(b) #adds directly to a and modifies it
print(a)

#Dot products
#A dot product is the sum of the products of the corresponding entries of two 1D tensors. 

a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype=torch.float)
print(a.dot(b))

#Matrix multiplication
a = torch.tensor([[0,2,4],[1,3,5]], dtype=torch.float)
b = torch.tensor([[6,7],[8,9],[10,11]], dtype=torch.float)

c = torch.mm(a,b)
print(c)

#Matrix multiplication with broadcasting
t1 = torch.randn(2, 3, 4)
t2 = torch.randn(4, 5)

print(torch.matmul(t1,t2))

#Advanced operations
#L2 or Euclidian Norm
x = torch.tensor([2.,5.,8.,14.])
x.norm()

#Number of elements
x = torch.ones(3,7)
x.numel()



















