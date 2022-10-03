import numpy as np 

class Memory:
  def __init__(self, max_size, img_size):
    memory_shape = [
            ('img', np.float32, img_size.shape), ('action', np.int64),
            ('reward', np.float32), ('next_img', np.float32, img_size.shape),
            ('done', np.float32)
        ]
    self.memory = np.zeros(max_size, dtype=memory_shape)
    self.max_size = max_size
    self.counter = 0 
    

  def save(self,img,action,reward, next_img,done):
    index = self.counter % self.max_size
    self.memory[index] = (img,action,reward,next_img,done)
    self.counter += 1
  
  def sample(self, batch_size):
    curr = min(self.counter, self.max_size)
    indices = np.random.choice(curr,size=batch_size,replace = False) #replace is false so samples are unique 
    batch_sample = self.memory[indices]

    return (np.array(batch_sample['img']),
            np.array(batch_sample['action']),
            np.array(batch_sample['reward']),
            np.array(batch_sample['next_img']),
            np.array(batch_sample['done']))