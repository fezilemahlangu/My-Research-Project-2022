import torch 
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import LazyConv2d
from torch.nn import LazyLinear 
from torch.nn import Linear #for fully connected layers
from torch.optim import Adam #optimizer 
import torch.nn.functional as F
import csv 
from torch import flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
  def __init__(self, numChannels, classes):
    
    super(Net, self).__init__()
    self.numChannels = numChannels

    self.conv1 = Conv2d(numChannels.shape[0],32, 8,stride=4)
    self.conv2 = LazyConv2d(64, kernel_size=4,stride=2)
    self.conv3 = LazyConv2d(64, kernel_size=3,stride=1)
    self.fc1 = LazyLinear(512)
    self.fc2 = LazyLinear(classes)

    self.lr = 1e-4
    self.optimizer = Adam(self.parameters(),lr=self.lr)

    self.loss = nn.MSELoss()

  def forward(self, x):

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = flatten(x,1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x

  def backward(self,target,val):
    self.optimizer.zero_grad()
    loss = self.loss(val,target).to(device)
    loss.backward()
    self.optimizer.step()
