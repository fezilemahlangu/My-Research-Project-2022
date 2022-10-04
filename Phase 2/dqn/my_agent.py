import torch 
import numpy as np 
import random 
from dqn.network import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_Agent:
    def __init__(self, img_size, num_classes, memory):
        self.img_size = img_size
        self.num_classes = num_classes
        self.gamma = 0.99 #discount factor 
        #self.memory = deque(maxlen=5000)
        self.epsilon = 1.0 #exploration rate start
        self.epsilon_decay = 0.1
        self.epsilon_min = 0.01 #exploration rate end 
        self.learning_rate = 0.001

        self.memory = memory
        self.model = Net(img_size,num_classes)
        self.target = Net(img_size,num_classes)

        self.batch_size = 32
        self.batch_space = [i for i in range(self.batch_size)]
        

    
    # def remember(self, img, action, reward, next_img, done):
    #     self.memory.save((img,action,reward, next_img, done))
        # print("remembered.")

    def train(self):
        img,action,reward,next_img,done = self.memory.sample(self.batch_size)

        img = img / 255.0
        next_img = next_img / 255.0

        # img = torch.tensor([img], dtype=torch.float)
        # img = img.to(device)

        # next_img = torch.tensor([next_img], dtype=torch.float)
        # next_img = next_img.to(device)

        img = torch.from_numpy(img).float().to(device)
        next_img = torch.from_numpy(next_img).float().to(device)
        action = torch.from_numpy(action).long().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        done = torch.from_numpy(done).float().to(device)
        
     
        action_val = self.model.forward(img)
        action_val = action_val[self.batch_space, action]

        action_val_next = self.target.forward(next_img)
        action_val_next = action_val_next.max(dim=1)[0]

        action_val_target = reward + (1 - done) * self.gamma * action_val_next
        

        #propagate errors and step 
        
        self.model.backward(action_val_target, action_val)



    def update_target(self): #update target network parameters with main parameters 
      
      self.target.load_state_dict(self.model.state_dict())

    def remember(self,img,action,reward, next_img,done):
      self.memory.save(img,action,reward, next_img,done)


    
    def act(self, img, fraction):
        

        eps = self.epsilon + fraction * (self.epsilon_min - self.epsilon)

        if np.random.rand() < eps:
            return random.randrange(self.num_classes) 

        img = np.array(img) / 255.0 #normalize 
        # img = torch.tensor([img], dtype=torch.float)
        # img = img.to(device)
        
        img = torch.from_numpy(img).float().unsqueeze(0).to(device) #unsqueeze so it's 4D with batch size dimension

        act_values = self.model.forward(img)
        # return np.argmax(act_values[0])
        return torch.argmax(act_values).item()