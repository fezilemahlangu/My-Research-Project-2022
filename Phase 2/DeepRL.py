import torch 
import numpy as np
import random 
import gym 

from dqn.wrappers import *
from dqn.memory_replay  import Memory
from dqn.my_agent import DQN_Agent

def main():
  #seed
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)

  #arguments 
  args = {
      "env_name" : "Breakout-v4", #name of environment used
      "learning_starts" :10000, 
      "learning_freq" : 1, 
      "update_target_freq" : 1000, 
      "memory_size" : 50000, 
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
  print(img_size.shape)
  # img_size = env.observation_space.shape
  num_classes = env.action_space.n

  memory = Memory(args['memory_size'],img_size)
  agent = DQN_Agent(img_size, num_classes, memory)

  state = env.reset()



  episode_rewards = [0.0]
  loss = [0.0]
  eps_timesteps = agent.epsilon_decay * float(args["n_episodes"])

  for e in range(args["n_episodes"]):
      # state = env.reset()
      # state = np.reshape(state, img_size)
      # state = np.reshape(state, (-1,)+img_size)

      fraction = min(1.0, float(e) / eps_timesteps)
    
      action = agent.act(state, fraction)
      # print(action)

      next_state, reward, done, _ = env.step(action) 
      # print(reward)
      # next_state = np.reshape(next_state, (-1,)+img_size)
      

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
          
    
      if done:
        print(f"steps: {e}")
        print(f"epsiodes: {len(episode_rewards)}")
        print(f"cumulative episode reward: {round(np.sum(episode_rewards),1)}")
        print("------------------------------------")

if __name__ == '__main__':
  main()