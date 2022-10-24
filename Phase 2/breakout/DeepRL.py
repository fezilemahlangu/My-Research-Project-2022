
import torch 
import numpy as np
import random 
import gym 
import csv 

from dqn.wrappers import *
from dqn.memory_replay  import Memory
from dqn.my_agent import DQN_Agent

def save_reward(fieldnames,rewards, model ):
  fieldnames = ['rewards']
  rows=[
      {
          'rewards' : rewards
      }
  ]

  with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/results_phase2.csv', 'a', encoding='UTF8') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writerows(rows)

  rows = [

    {
      'models' : model 
    }
  ]

  fieldnames = ['models']

  with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/models_phase2.csv', 'a', encoding='UTF8') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writerows(rows)

def main():
  #seed
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)

  #prepare csv 
  fieldnames_results = ['rewards'] 
  with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/results_phase2.csv', 'w', encoding='UTF8', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames_results)
      writer.writeheader()

  fieldnames_results = ['models'] 
  with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/models_phase2.csv', 'w', encoding='UTF8', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames_results)
      writer.writeheader()

  #arguments 
  args = {
      "env_name" : "BreakoutNoFrameskip-v4", #name of environment used #Enduro-v4
      "learning_starts" :10000, 
      "learning_freq" : 5, 
      "update_target_freq" : 1000, 
      "batch_size" : 32,
      "gamma" : 0.99,
      "memory_size" : 5000, 
      "eps_start" : 1.0,
      "eps_end" : 0.01,
      "eps_fraction" : 0.1,
      "print_freq" : 10,
      "n_episodes" : int(1e6) #number of steps the environment will run 
  }

  
  
  #-------------hyperparams-------------#
  params = []
  params.append([[16, 1, 1, 2, 0.05, 32, 0.05, 0.001],[],[]]) #size, kernel,stride , linear_size,learning_rate
  params.append([[16, 7, 1, 2, 0.05, 256, 0.05, 0.003],[32, 3, 1, 3, 0.25],[]])
  params.append([[16, 7, 4, 1, 0.05, 256, 0.05, 0.001],[32, 4, 2,  1, 0.25],[64, 3, 1,  1, 0.25]])
  # params.append([[32, 3, 1, 2, 0.05, 256, 0.05, 0.0001],[16, 1, 1, 2, 0.15],[]])
  # params.append([[32, 3, 2, 2, 0.05, 128, 0.05, 1e-15],[128, 3, 2, 3, 0.25],[]])
  
  # params.append([[16, 7, 4, 1, 0.05, 256, 0.05, 0.001],[32, 4, 2, 1, 0.25],[64, 3, 1,1, 0.25]]) #size, kernel,stride 
  # params.append([[16,5,4,1e-4,512],[32,4,2],[64,3,1]]) #size, kernel,stride 
  # params.append([[32,8,4,1e-4,512],[64,4,2],[128,3,1]]) #size, kernel,stride
  #-------------------------------------#

  for p in params:
      first = p[0]
      second = p[1]
      third = p[2]

      env = gym.make(args["env_name"])
      # env.reset()
      env.seed(42)

      #--------------wrappers---------------#
      env = NoopResetEnv(env, noop_max=30)
      env = MaxAndSkipEnv(env, skip=4)
      env = EpisodicLifeEnv(env)
      env = FireResetEnv(env)
      env = WarpFrame(env)
      env = PyTorchFrame(env)
      #env = ClipRewardEnv(env)
      env = FrameStack(env, 4)
      #-------------------------------------#

      img_size = env.observation_space
      # print(img_size.shape)

      num_classes = env.action_space.n


      memory = Memory(args['memory_size'])
      
      agent = DQN_Agent(img_size, num_classes, memory,args["batch_size"], args["gamma"],first,second,third)

      state = env.reset()

      episode_rewards = [0.0]
      loss = [0.0]
      mean_rewards = []
      # episodes = []
      eps_timesteps = args["eps_fraction"] * float(args["n_episodes"])

      for e in range(args["n_episodes"]):

          fraction = min(1.0, float(e) / eps_timesteps)

          eps_threshold = args["eps_start"] + fraction *(args["eps_end"] - args["eps_start"])

          sample = random.random()

          if sample < eps_threshold:
            action = random.randrange(env.action_space.n)
          else:
            action = agent.act(state)

          next_state, reward, done, _ = env.step(action) 

          agent.replay_buffer.add(state,action,reward,next_state,float(done))
          state = next_state

          episode_rewards[-1] += reward 
          if done:
            state = env.reset()
            episode_rewards.append(0.0)

          if e > args["learning_starts"]:
            if(e % args["learning_freq"] == 0):
              #train network
              agent.train()
            
            if (e % args["update_target_freq"] == 0):
              #update target network 
              agent.update_target()
              
        
          if (done and len(episode_rewards) % args["print_freq"] == 0):
            print(f"steps: {e}")
            print(f"epsiodes: {len(episode_rewards)}")
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 2)
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            mean_rewards.append(mean_100ep_reward)
            print("------------------------------------")
      save_reward(fieldnames_results,mean_rewards, p)

if __name__ == '__main__':
  main()