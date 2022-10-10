from ast import arg
import torch 
import numpy as np
import random 
import gym 
import csv 

from dqn.wrappers import *
from dqn.memory_replay  import Memory
from dqn.my_agent import DQN_Agent

def save_reward(fieldnames,rewards):
  rows=[
      {
          'rewards' : rewards
      }
  ]


  with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/results_phase2.csv', 'a', encoding='UTF8') as f:
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

  #arguments 
  args = {
      "env_name" : "Breakout-v4", #name of environment used
      "learning_starts" :10000, 
      "learning_freq" : 5, 
      "update_target_freq" : 1000, 
      "memory_size" : 50000, 
      "print_freq" : 10,
      "n_episodes" : int(1e6) #number of steps the environment will run 
  }

  env = gym.make(args["env_name"],full_action_space=True)
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

  memory = Memory(args['memory_size'],img_size)
  agent = DQN_Agent(img_size, num_classes, memory)

  state = env.reset()


  episode_rewards = [0.0]
  loss = [0.0]
  mean_rewards = []
  # episodes = []
  eps_timesteps = agent.epsilon_decay * float(args["n_episodes"])

  for e in range(args["n_episodes"]):

      fraction = min(1.0, float(e) / eps_timesteps)
    
      action = agent.act(state, fraction)

      next_state, reward, done, _ = env.step(action) 

      agent.remember(state,action,reward,next_state,float(done))
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
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        mean_rewards.append(mean_100ep_reward)
        print("------------------------------------")
  save_reward(fieldnames_results,mean_rewards)

if __name__ == '__main__':
  main()