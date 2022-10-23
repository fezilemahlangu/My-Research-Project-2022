import torch 
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d 
from torch.nn import Dropout
from torch.nn import Linear #for fully connected layers
# from torch.optim import Adam #optimizer 
import torch.nn.functional as F
# import csv 
import numpy as np 
from torch import flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
  def __init__(self, numChannels, classes,first,second,third):
    
    super(Net, self).__init__()
    self.numChannels = numChannels
    self.classes = classes
    self.first = first
    self.second = second 
    self.third = third

    #first = [conv_size, kernel_size, stride_size, maxpool_size, dropout_rate, dense_size, dropout_rate, learnig_rate]
    #second = [conv_size, kernel_size, stride_size,maxpool_size,dropout_rate]
    #third = [conv_size, kernel_size, stride_size,maxpool_size,dropout_rate]

    self.conv1 = Conv2d(numChannels.shape[0],self.first[0], self.first[1],self.first[2])
                                              #conv size,   kernel_size,  stride_size
    self.maxpool1 = MaxPool2d(self.first[3]) #maxpool size 
    self.dropout1 = Dropout(self.first[4]) #dropout

    self.conv2 = 0
    self.maxpool2 = 0
    self.dropout2 = 0

    self.conv3 = 0
    self.maxpool3 = 0
    self.dropout3 = 0
  
    

    if len(self.second)>1:
        self.conv2 = Conv2d(self.first[0],self.second[0], self.second[1],self.second[2])
        self.maxpool2 = MaxPool2d(self.second[3])
        self.dropout2 = Dropout(self.second[4])
    if len(self.third)>1:
        
        self.conv3 = Conv2d(self.second[0],self.third[0],self.third[1],self.third[2])
        self.maxpool3 = MaxPool2d(self.third[3])
        self.dropout3 = Dropout(self.third[4])

    flatshape = self.flat_shape(self.numChannels.shape)

    self.fc1 = Linear(flatshape,self.first[5]) #dense size
    self.dropout3 =  Dropout(self.first[6]) #dropout rate 
    self.fc2 = Linear(self.first[5],self.classes)

    # self.lr = self.first[7]
    # self.optimizer = Adam(self.parameters(),lr=self.lr)

    # self.loss = nn.functional.l1_loss()

  def flat_shape(self,input):
    x = torch.zeros(1,*input)
    x = self.conv1(x)
    x = self.maxpool1(x)

    if len(self.second)>1:
      x = self.conv2(x)
      x = self.maxpool2(x)

    if len(self.third)>1:
      x = self.conv3(x)
      x = self.maxpool3(x)

    return int(np.prod(x.size()))

  def forward(self, x):

    x = F.relu(self.conv1(x))
    x = self.maxpool1(x)
    x = self.dropout1(x)

    if len(self.second)>1:
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

    if len(self.third)>1:
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)

    x = flatten(x,1)
    x = F.relu(self.fc1(x))
    x = self.dropout3(x)
    x = self.fc2(x)

    return x

  # def backward(self,target,val):
  #   self.optimizer.zero_grad()
  #   loss = nn.functional.l1_loss(val,target).to(device)
  #   loss.backward()
  #   self.optimizer.step()
