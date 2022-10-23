import torch 
from torch.optim import Adam
from torch import nn
from gym import spaces 
import numpy as np 
# import random 
from dqn.network import Net
from dqn.memory_replay import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_Agent:

    def __init__(
    self,
    observation_space: spaces.Box,
    action_space: spaces.Discrete,
    replay_buffer: Memory,
    batch_size,
    gamma,
    first,
    second,
    third
    ):
        self.lr = first[7]
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        #agents networks
        self.target_network = Net(observation_space,action_space,first,second,third).to(device)
        self.policy_network = Net(observation_space,action_space,first,second,third).to(device)
        #agents optimiser 
        self.optimiser = Adam(self.policy_network.parameters(),lr=self.lr)
    

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        tensor_states = torch.from_numpy(states/255.0).float().to(device)
        tensor_next_states = torch.from_numpy(next_states/255.0).float().to(device)
        tensor_actions = torch.from_numpy(actions).long().to(device)
        tensor_rewards = torch.from_numpy(rewards).float().to(device)
        tensor_dones = torch.from_numpy(dones).float().to(device)

        # # don't track gradients
        # with torch.no_grad():
        #     # calculate target
        #     estimate = None
        #     if self.use_double_dqn:
        #         # get next action from the other network
        #         _, next_action = self.policy_network(tensor_states).max(1)
        #         # get next q from target network
        #         estimate = self.target_network(tensor_next_states).gather(1, next_action.unsqueeze(1)).squeeze()
        #     else:
        #         estimate = None
              
        #         # get next q from target network
                

        estimate = self.target_network(tensor_next_states).max(1)

        c = self.gamma * estimate[0]
        target = tensor_rewards + (1 - tensor_dones) * c

        # backpropagation
        new_estimate = self.policy_network(tensor_states).gather(1, tensor_actions.unsqueeze(1)).squeeze()
        loss = nn.functional.l1_loss(new_estimate, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # # remove from GPU
        # del tensor_states
        # del tensor_next_states

        return loss.item()


    def update_target(self): #update target network parameters with main parameters 
      
      self.target_network.load_state_dict(self.policy_network.state_dict())

    # def remember(self,img,action,reward, next_img,done):
    #   self.memory.save(img,action,reward, next_img,done)


    
    def act(self, state):
        

        state = np.array(state)/255.0
        # convert state to tensor object and put on GPU
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # making sure gradients aren't saved for the following calculations
        with torch.no_grad():
            # get action-state values using the foward pass of the network
            qs = self.policy_network(state)
            # get max action
            _, action = qs.max(1)
            # return action from tensor object
            return action.item()