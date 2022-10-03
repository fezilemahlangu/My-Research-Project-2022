import torch 
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Linear #for fully connected layers
from torch.optim import Adam #optimizer 
import torch.nn.functional as F
import csv 


device = torch.device("cpu")
class Net(nn.Module):
  def __init__(self, numChannels, classes):
    
    super(Net, self).__init__()
    # self.conv1 = Conv2d(numChannels.shape[0],32, 3,stride=4)
    # self.maxpool1 = MaxPool2d(kernel_size=(1, 1), stride=(2, 2))
    # self.conv2 = Conv2d(32, 64, kernel_size=1,stride=2)
    # self.maxpool2 = MaxPool2d(kernel_size=1, stride=(2, 2))
    # self.fc1 = Linear(in_features=320, out_features=100)
    # self.fc2 = Linear(in_features=100, out_features=classes)
    self.numChannels = numChannels

    self.conv1 = Conv2d(numChannels.shape[0],32, 8,stride=4)
    #self.maxpool1 = MaxPool2d(kernel_size=(1, 1), stride=(2, 2))
    self.conv2 = Conv2d(32, 64, kernel_size=4,stride=2)
    self.conv3 = Conv2d(64, 64, kernel_size=3,stride=1)
    #self.maxpool2 = MaxPool2d(kernel_size=1, stride=(2, 2))
    self.fc1 = Linear(in_features=64*7*7, out_features=512)
    self.fc2 = Linear(in_features=512, out_features=classes)


    self.lr = 1e-4
    self.optimizer = Adam(self.parameters(),lr=self.lr)

    self.loss = nn.MSELoss()

  def forward(self, x):
   
    # x = F.relu(self.conv1(x))
    # x = self.maxpool1(x)
    # x = F.relu(self.conv2(x))
    # x = self.maxpool2(x)
    # x = flatten(x, 1)
    # x = F.relu(self.fc1(x))
    # x = self.fc2(x)

    x = F.relu(self.conv1(x))
    #x = self.maxpool1(x)
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 64 * 7 * 7)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x

  def backward(self,target,val):
    self.optimizer.zero_grad()
    loss = self.loss(val,target).to(device)
    loss.backward()
    self.optimizer.step()
